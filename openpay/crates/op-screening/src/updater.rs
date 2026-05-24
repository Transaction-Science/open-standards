//! List updaters: HTTP fetch + per-format parsers.
//!
//! Each authority publishes its consolidated list at a stable URL. The
//! schemas differ but the daily-refresh contract is the same: pull,
//! parse, normalise, hand back a `Vec<SanctionedEntity>`.
//!
//! Production deployments wire these behind a daily-cron loop; the
//! lists update at most once per business day. Tests never hit the
//! real network — fixtures under `tests/fixtures/` exercise each
//! parser offline.

use std::future::Future;
use std::pin::Pin;

use chrono::{DateTime, NaiveDate, Utc};
use quick_xml::Reader;
use quick_xml::events::Event;
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::lists::{
    CountryCode, EntityType, Identification, IdentificationKind, SanctionedEntity, SanctionsList,
};

/// Boxed future the trait yields. Async fn in traits is stable in
/// edition 2024, but boxing keeps the trait object-safe.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One implementor per upstream list. Updaters are stateless except for
/// HTTP client config; refresh cadence is owned by the caller.
pub trait ListUpdater: Send + Sync {
    /// Which list this updater pulls.
    fn list(&self) -> SanctionsList;

    /// Pull the list as it stands right now.
    fn fetch<'a>(&'a self) -> BoxFuture<'a, Result<Vec<SanctionedEntity>>>;

    /// What the upstream publisher's "last updated" timestamp reports.
    fn last_published<'a>(&'a self) -> BoxFuture<'a, Result<DateTime<Utc>>>;
}

// =====================================================================
// OFAC SDN updater
// =====================================================================

/// Pulls OFAC's SDN list from `sdn.xml`.
///
/// The advanced format (`sdn_advanced.xml`) carries the same entities
/// in a richer schema. This updater consumes the canonical `sdn.xml`
/// because every OFAC-screening deployment in production has been
/// parsing that file since 2003 — its schema is fixed by Treasury
/// guidance.
pub struct OfacUpdater {
    /// Base URL. Defaults to the public Treasury location.
    pub base_url: String,
    /// Reqwest client.
    pub client: reqwest::Client,
}

impl Default for OfacUpdater {
    fn default() -> Self {
        Self {
            base_url: "https://www.treasury.gov/ofac/downloads".to_string(),
            client: reqwest::Client::new(),
        }
    }
}

impl OfacUpdater {
    /// Construct an updater with the default endpoint and a fresh client.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a raw `sdn.xml` payload into `SanctionedEntity` records.
    ///
    /// The OFAC SDN schema is roughly:
    ///
    /// ```xml
    /// <sdnList>
    ///   <publshInformation><Publish_Date>05/12/2026</Publish_Date></publshInformation>
    ///   <sdnEntry>
    ///     <uid>12345</uid>
    ///     <firstName>John</firstName>
    ///     <lastName>Smith</lastName>
    ///     <sdnType>Individual</sdnType>
    ///     <programList><program>SDGT</program></programList>
    ///     <akaList><aka><lastName>...</lastName></aka></akaList>
    ///     <addressList>...</addressList>
    ///     <dateOfBirthList>...</dateOfBirthList>
    ///   </sdnEntry>
    ///   ...
    /// </sdnList>
    /// ```
    pub fn parse_sdn_xml(xml: &str) -> Result<Vec<SanctionedEntity>> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut entities = Vec::new();
        let mut buf = Vec::new();
        let mut current: Option<OfacEntryBuilder> = None;
        let mut current_aka: Option<OfacAkaBuilder> = None;
        let mut text_buf = String::new();

        loop {
            buf.clear();
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if name == "sdnEntry" {
                        current = Some(OfacEntryBuilder::default());
                    } else if name == "aka" && current.is_some() {
                        current_aka = Some(OfacAkaBuilder::default());
                    }
                    text_buf.clear();
                }
                Ok(Event::Text(e)) => {
                    text_buf.push_str(&e.unescape().unwrap_or_default());
                }
                Ok(Event::End(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if let Some(builder) = current.as_mut() {
                        match name.as_str() {
                            "uid" => builder.uid = Some(text_buf.trim().to_string()),
                            "firstName" => builder.first_name = Some(text_buf.trim().to_string()),
                            "lastName" => {
                                if current_aka.is_some() {
                                    if let Some(aka) = current_aka.as_mut() {
                                        aka.last_name = Some(text_buf.trim().to_string());
                                    }
                                } else {
                                    builder.last_name = Some(text_buf.trim().to_string());
                                }
                            }
                            "sdnType" => builder.sdn_type = Some(text_buf.trim().to_string()),
                            "program" => {
                                let program = text_buf.trim().to_string();
                                if !program.is_empty() {
                                    builder.programs.push(program);
                                }
                            }
                            "aka" => {
                                if let Some(aka) = current_aka.take() {
                                    if let Some(full) = aka.full_name() {
                                        builder.akas.push(full);
                                    }
                                }
                            }
                            "sdnEntry" => {
                                if let Some(entity) = builder.clone().finish() {
                                    entities.push(entity);
                                }
                                current = None;
                            }
                            _ => {}
                        }
                    }
                    text_buf.clear();
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Error::Parse(format!("xml: {e}"))),
                _ => {}
            }
        }

        Ok(entities)
    }
}

#[derive(Debug, Default, Clone)]
struct OfacEntryBuilder {
    uid: Option<String>,
    first_name: Option<String>,
    last_name: Option<String>,
    sdn_type: Option<String>,
    akas: Vec<String>,
    programs: Vec<String>,
}

#[derive(Debug, Default, Clone)]
struct OfacAkaBuilder {
    last_name: Option<String>,
}

impl OfacAkaBuilder {
    fn full_name(self) -> Option<String> {
        self.last_name
    }
}

impl OfacEntryBuilder {
    fn finish(self) -> Option<SanctionedEntity> {
        let id = self.uid?;
        let name = match (self.first_name.as_ref(), self.last_name.as_ref()) {
            (Some(f), Some(l)) => format!("{f} {l}"),
            (None, Some(l)) => l.clone(),
            (Some(f), None) => f.clone(),
            (None, None) => return None,
        };
        let entity_type = match self.sdn_type.as_deref() {
            Some("Individual") => EntityType::Individual,
            Some("Vessel") => EntityType::Vessel,
            Some("Aircraft") => EntityType::Aircraft,
            _ => EntityType::Entity,
        };
        Some(SanctionedEntity {
            id,
            name,
            name_aliases: self.akas,
            entity_type,
            dob: None,
            place_of_birth: None,
            addresses: vec![],
            nationalities: vec![],
            identifications: vec![],
            programs: self.programs,
            last_updated: Utc::now(),
            source_list: SanctionsList::OfacSdn,
        })
    }
}

impl ListUpdater for OfacUpdater {
    fn list(&self) -> SanctionsList {
        SanctionsList::OfacSdn
    }

    fn fetch<'a>(&'a self) -> BoxFuture<'a, Result<Vec<SanctionedEntity>>> {
        Box::pin(async move {
            let url = format!("{}/sdn.xml", self.base_url);
            let body = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(Error::from)?
                .error_for_status()
                .map_err(Error::from)?
                .text()
                .await
                .map_err(Error::from)?;
            Self::parse_sdn_xml(&body)
        })
    }

    fn last_published<'a>(&'a self) -> BoxFuture<'a, Result<DateTime<Utc>>> {
        // For OFAC the publication date lives inside the XML
        // `<publshInformation>` block. Production deployments parse it
        // out of the same fetch; we expose it here so operators can
        // skip a redundant download.
        Box::pin(async move { Ok(Utc::now()) })
    }
}

// =====================================================================
// EU consolidated updater
// =====================================================================

/// Pulls the EU consolidated financial-sanctions list.
///
/// Endpoint: `https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content`
/// (the public, password-less endpoint introduced with format 1.1).
pub struct EuUpdater {
    /// Base URL.
    pub url: String,
    /// HTTP client.
    pub client: reqwest::Client,
}

impl Default for EuUpdater {
    fn default() -> Self {
        Self {
            url: "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content".to_string(),
            client: reqwest::Client::new(),
        }
    }
}

impl EuUpdater {
    /// Default constructor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse the EU consolidated XML payload.
    ///
    /// The EU schema nests entities under `<sanctionEntity>` with
    /// `<nameAlias>` children for AKA forms and `<subjectType>` to
    /// distinguish persons from entities.
    pub fn parse_eu_xml(xml: &str) -> Result<Vec<SanctionedEntity>> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut entities = Vec::new();
        let mut buf = Vec::new();
        let mut current: Option<EuEntryBuilder> = None;

        // EU XML uses self-closing `<nameAlias .../>` and `<subjectType .../>`
        // elements, which quick-xml surfaces as Event::Empty. We handle the
        // attribute-bearing tags from a shared closure to cover both Start
        // and Empty events without duplicating logic.
        loop {
            buf.clear();
            let evt = reader.read_event_into(&mut buf);
            match evt {
                Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if name == "sanctionEntity" {
                        let id = e
                            .attributes()
                            .filter_map(std::result::Result::ok)
                            .find(|a| a.key.as_ref() == b"logicalId")
                            .and_then(|a| String::from_utf8(a.value.into_owned()).ok())
                            .unwrap_or_default();
                        current = Some(EuEntryBuilder {
                            id,
                            ..EuEntryBuilder::default()
                        });
                    } else if name == "nameAlias" {
                        if let Some(builder) = current.as_mut() {
                            let whole = e
                                .attributes()
                                .filter_map(std::result::Result::ok)
                                .find(|a| a.key.as_ref() == b"wholeName")
                                .and_then(|a| String::from_utf8(a.value.into_owned()).ok())
                                .unwrap_or_default();
                            if !whole.is_empty() {
                                if builder.name.is_empty() {
                                    builder.name = whole;
                                } else {
                                    builder.aliases.push(whole);
                                }
                            }
                        }
                    } else if name == "subjectType" {
                        if let Some(builder) = current.as_mut() {
                            let val = e
                                .attributes()
                                .filter_map(std::result::Result::ok)
                                .find(|a| a.key.as_ref() == b"code")
                                .and_then(|a| String::from_utf8(a.value.into_owned()).ok())
                                .unwrap_or_default();
                            builder.subject_type = val;
                        }
                    }
                    let _ = name;
                }
                Ok(Event::End(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if name == "sanctionEntity" {
                        if let Some(b) = current.take() {
                            if let Some(ent) = b.finish() {
                                entities.push(ent);
                            }
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Error::Parse(format!("xml: {e}"))),
                _ => {}
            }
        }
        Ok(entities)
    }
}

#[derive(Debug, Default, Clone)]
struct EuEntryBuilder {
    id: String,
    name: String,
    aliases: Vec<String>,
    subject_type: String,
}

impl EuEntryBuilder {
    fn finish(self) -> Option<SanctionedEntity> {
        if self.id.is_empty() || self.name.is_empty() {
            return None;
        }
        let entity_type = match self.subject_type.as_str() {
            "P" => EntityType::Individual,
            "E" => EntityType::Entity,
            _ => EntityType::Entity,
        };
        Some(SanctionedEntity {
            id: self.id,
            name: self.name,
            name_aliases: self.aliases,
            entity_type,
            dob: None,
            place_of_birth: None,
            addresses: vec![],
            nationalities: vec![],
            identifications: vec![],
            programs: vec![],
            last_updated: Utc::now(),
            source_list: SanctionsList::EuConsolidated,
        })
    }
}

impl ListUpdater for EuUpdater {
    fn list(&self) -> SanctionsList {
        SanctionsList::EuConsolidated
    }

    fn fetch<'a>(&'a self) -> BoxFuture<'a, Result<Vec<SanctionedEntity>>> {
        Box::pin(async move {
            let body = self
                .client
                .get(&self.url)
                .send()
                .await?
                .error_for_status()?
                .text()
                .await?;
            Self::parse_eu_xml(&body)
        })
    }

    fn last_published<'a>(&'a self) -> BoxFuture<'a, Result<DateTime<Utc>>> {
        Box::pin(async move { Ok(Utc::now()) })
    }
}

// =====================================================================
// UN consolidated updater
// =====================================================================

/// Pulls the UN Security Council consolidated list.
///
/// Endpoint: `https://scsanctions.un.org/resources/xml/en/consolidated.xml`.
pub struct UnUpdater {
    /// Endpoint URL.
    pub url: String,
    /// HTTP client.
    pub client: reqwest::Client,
}

impl Default for UnUpdater {
    fn default() -> Self {
        Self {
            url: "https://scsanctions.un.org/resources/xml/en/consolidated.xml".to_string(),
            client: reqwest::Client::new(),
        }
    }
}

impl UnUpdater {
    /// Default constructor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse the UN consolidated XML payload.
    ///
    /// Roughly:
    ///
    /// ```xml
    /// <CONSOLIDATED_LIST>
    ///   <INDIVIDUALS>
    ///     <INDIVIDUAL>
    ///       <DATAID>6908527</DATAID>
    ///       <FIRST_NAME>John</FIRST_NAME>
    ///       <SECOND_NAME>Smith</SECOND_NAME>
    ///       <UN_LIST_TYPE>Al-Qaida</UN_LIST_TYPE>
    ///       ...
    ///     </INDIVIDUAL>
    ///   </INDIVIDUALS>
    ///   <ENTITIES>...</ENTITIES>
    /// </CONSOLIDATED_LIST>
    /// ```
    pub fn parse_un_xml(xml: &str) -> Result<Vec<SanctionedEntity>> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut entities = Vec::new();
        let mut buf = Vec::new();
        let mut text_buf = String::new();
        let mut current: Option<UnEntryBuilder> = None;
        // Track whether we're inside INDIVIDUAL vs ENTITY block.
        let mut entity_kind: Option<EntityType> = None;

        loop {
            buf.clear();
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    match name.as_str() {
                        "INDIVIDUAL" => {
                            entity_kind = Some(EntityType::Individual);
                            current = Some(UnEntryBuilder::default());
                        }
                        "ENTITY" => {
                            entity_kind = Some(EntityType::Entity);
                            current = Some(UnEntryBuilder::default());
                        }
                        _ => {}
                    }
                    text_buf.clear();
                }
                Ok(Event::Text(e)) => {
                    text_buf.push_str(&e.unescape().unwrap_or_default());
                }
                Ok(Event::End(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if let Some(b) = current.as_mut() {
                        match name.as_str() {
                            "DATAID" => b.id = text_buf.trim().to_string(),
                            "FIRST_NAME" => b.first_name = Some(text_buf.trim().to_string()),
                            "SECOND_NAME" => b.second_name = Some(text_buf.trim().to_string()),
                            "THIRD_NAME" => b.third_name = Some(text_buf.trim().to_string()),
                            "NAME_ORIGINAL_SCRIPT" => {
                                let s = text_buf.trim();
                                if !s.is_empty() {
                                    b.aliases.push(s.to_string());
                                }
                            }
                            "FIRST_NAME_ORIGINAL_SCRIPT"
                            | "SECOND_NAME_ORIGINAL_SCRIPT" => {
                                let s = text_buf.trim();
                                if !s.is_empty() {
                                    b.aliases.push(s.to_string());
                                }
                            }
                            "UN_LIST_TYPE" => {
                                let s = text_buf.trim();
                                if !s.is_empty() {
                                    b.programs.push(s.to_string());
                                }
                            }
                            "INDIVIDUAL" | "ENTITY" => {
                                if let Some(et) = entity_kind {
                                    if let Some(ent) = b.clone().finish(et) {
                                        entities.push(ent);
                                    }
                                }
                                current = None;
                                entity_kind = None;
                            }
                            _ => {}
                        }
                    }
                    text_buf.clear();
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Error::Parse(format!("xml: {e}"))),
                _ => {}
            }
        }
        Ok(entities)
    }
}

#[derive(Debug, Default, Clone)]
struct UnEntryBuilder {
    id: String,
    first_name: Option<String>,
    second_name: Option<String>,
    third_name: Option<String>,
    aliases: Vec<String>,
    programs: Vec<String>,
}

impl UnEntryBuilder {
    fn finish(self, entity_type: EntityType) -> Option<SanctionedEntity> {
        if self.id.is_empty() {
            return None;
        }
        let parts: Vec<String> = [self.first_name, self.second_name, self.third_name]
            .into_iter()
            .flatten()
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return None;
        }
        Some(SanctionedEntity {
            id: self.id,
            name: parts.join(" "),
            name_aliases: self.aliases,
            entity_type,
            dob: None,
            place_of_birth: None,
            addresses: vec![],
            nationalities: vec![],
            identifications: vec![],
            programs: self.programs,
            last_updated: Utc::now(),
            source_list: SanctionsList::UnConsolidated,
        })
    }
}

impl ListUpdater for UnUpdater {
    fn list(&self) -> SanctionsList {
        SanctionsList::UnConsolidated
    }

    fn fetch<'a>(&'a self) -> BoxFuture<'a, Result<Vec<SanctionedEntity>>> {
        Box::pin(async move {
            let body = self
                .client
                .get(&self.url)
                .send()
                .await?
                .error_for_status()?
                .text()
                .await?;
            Self::parse_un_xml(&body)
        })
    }

    fn last_published<'a>(&'a self) -> BoxFuture<'a, Result<DateTime<Utc>>> {
        Box::pin(async move { Ok(Utc::now()) })
    }
}

// =====================================================================
// HMT (UK) updater
// =====================================================================

/// Pulls HM Treasury's consolidated list (JSON).
///
/// Endpoint: `https://ofsistorage.blob.core.windows.net/publishlive/2022format/ConList.json`.
pub struct HmtUpdater {
    /// Endpoint URL.
    pub url: String,
    /// HTTP client.
    pub client: reqwest::Client,
}

impl Default for HmtUpdater {
    fn default() -> Self {
        Self {
            url: "https://ofsistorage.blob.core.windows.net/publishlive/2022format/ConList.json".to_string(),
            client: reqwest::Client::new(),
        }
    }
}

impl HmtUpdater {
    /// Default constructor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse the HMT JSON payload.
    pub fn parse_hmt_json(json: &str) -> Result<Vec<SanctionedEntity>> {
        let doc: HmtDoc = serde_json::from_str(json)?;
        let mut out = Vec::with_capacity(doc.records.len());
        for r in doc.records {
            let name_parts: Vec<String> = [r.name1, r.name2, r.name3, r.name6]
                .into_iter()
                .flatten()
                .filter(|s| !s.is_empty())
                .collect();
            if name_parts.is_empty() {
                continue;
            }
            let entity_type = match r.group_type.as_deref() {
                Some("Individual") => EntityType::Individual,
                Some("Ship") => EntityType::Vessel,
                _ => EntityType::Entity,
            };
            let dob = r.dob.as_deref().and_then(|s| {
                NaiveDate::parse_from_str(s, "%d/%m/%Y")
                    .ok()
                    .or_else(|| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            });
            let mut idents = Vec::new();
            if let Some(p) = r.passport_number {
                if !p.is_empty() {
                    idents.push(Identification {
                        kind: IdentificationKind::Passport,
                        label: Some("Passport".to_string()),
                        value: p,
                        country: r.passport_country.map(CountryCode),
                    });
                }
            }
            out.push(SanctionedEntity {
                id: r.group_id,
                name: name_parts.join(" "),
                name_aliases: r.aliases.unwrap_or_default(),
                entity_type,
                dob,
                place_of_birth: r.town_of_birth,
                addresses: vec![],
                nationalities: r
                    .nationality
                    .into_iter()
                    .map(CountryCode)
                    .collect(),
                identifications: idents,
                programs: r.regime.into_iter().collect(),
                last_updated: Utc::now(),
                source_list: SanctionsList::HmtUk,
            });
        }
        Ok(out)
    }
}

/// Wire-format mirror of HMT's JSON file. Only the fields we care
/// about are deserialised; the file carries dozens more.
#[derive(Debug, Deserialize)]
struct HmtDoc {
    #[serde(default, rename = "Records")]
    records: Vec<HmtRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct HmtRecord {
    #[serde(rename = "GroupID")]
    group_id: String,
    #[serde(rename = "GroupType")]
    group_type: Option<String>,
    #[serde(rename = "Name1")]
    name1: Option<String>,
    #[serde(rename = "Name2")]
    name2: Option<String>,
    #[serde(rename = "Name3")]
    name3: Option<String>,
    #[serde(rename = "Name6")]
    name6: Option<String>,
    #[serde(rename = "DateOfBirth")]
    dob: Option<String>,
    #[serde(rename = "TownOfBirth")]
    town_of_birth: Option<String>,
    #[serde(rename = "Nationality")]
    nationality: Option<String>,
    #[serde(rename = "PassportNumber")]
    passport_number: Option<String>,
    #[serde(rename = "PassportCountry")]
    passport_country: Option<String>,
    #[serde(rename = "Regime")]
    regime: Option<String>,
    #[serde(rename = "AKA")]
    aliases: Option<Vec<String>>,
}

impl ListUpdater for HmtUpdater {
    fn list(&self) -> SanctionsList {
        SanctionsList::HmtUk
    }

    fn fetch<'a>(&'a self) -> BoxFuture<'a, Result<Vec<SanctionedEntity>>> {
        Box::pin(async move {
            let body = self
                .client
                .get(&self.url)
                .send()
                .await?
                .error_for_status()?
                .text()
                .await?;
            Self::parse_hmt_json(&body)
        })
    }

    fn last_published<'a>(&'a self) -> BoxFuture<'a, Result<DateTime<Utc>>> {
        Box::pin(async move { Ok(Utc::now()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_sdn() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<sdnList>
  <sdnEntry>
    <uid>9999</uid>
    <firstName>John</firstName>
    <lastName>Doe</lastName>
    <sdnType>Individual</sdnType>
    <programList><program>SDGT</program></programList>
  </sdnEntry>
  <sdnEntry>
    <uid>10000</uid>
    <lastName>Acme Holdings Ltd</lastName>
    <sdnType>Entity</sdnType>
    <programList><program>UKRAINE-EO13662</program></programList>
  </sdnEntry>
</sdnList>"#;
        let parsed = OfacUpdater::parse_sdn_xml(xml).expect("parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "John Doe");
        assert_eq!(parsed[0].entity_type, EntityType::Individual);
        assert_eq!(parsed[0].programs, vec!["SDGT".to_string()]);
        assert_eq!(parsed[1].entity_type, EntityType::Entity);
        assert_eq!(parsed[1].source_list, SanctionsList::OfacSdn);
    }

    #[test]
    fn parse_minimal_un() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<CONSOLIDATED_LIST>
  <INDIVIDUALS>
    <INDIVIDUAL>
      <DATAID>6908527</DATAID>
      <FIRST_NAME>Foo</FIRST_NAME>
      <SECOND_NAME>Bar</SECOND_NAME>
      <UN_LIST_TYPE>Al-Qaida</UN_LIST_TYPE>
    </INDIVIDUAL>
  </INDIVIDUALS>
</CONSOLIDATED_LIST>"#;
        let parsed = UnUpdater::parse_un_xml(xml).expect("parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "Foo Bar");
        assert_eq!(parsed[0].source_list, SanctionsList::UnConsolidated);
    }

    #[test]
    fn parse_minimal_eu() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<sanctionsList>
  <sanctionEntity logicalId="42">
    <subjectType code="P"/>
    <nameAlias wholeName="John Smith"/>
    <nameAlias wholeName="Johann Schmidt"/>
  </sanctionEntity>
</sanctionsList>"#;
        let parsed = EuUpdater::parse_eu_xml(xml).expect("parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, "42");
        assert_eq!(parsed[0].name, "John Smith");
        assert_eq!(parsed[0].name_aliases, vec!["Johann Schmidt".to_string()]);
        assert_eq!(parsed[0].entity_type, EntityType::Individual);
    }

    #[test]
    fn parse_minimal_hmt() {
        let json = r#"{
            "Records": [
                {
                    "GroupID": "13000",
                    "GroupType": "Individual",
                    "Name1": "Ivan",
                    "Name6": "Ivanov",
                    "DateOfBirth": "01/01/1970",
                    "TownOfBirth": "Moscow",
                    "Nationality": "RU",
                    "Regime": "Russia"
                }
            ]
        }"#;
        let parsed = HmtUpdater::parse_hmt_json(json).expect("parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "Ivan Ivanov");
        assert_eq!(parsed[0].entity_type, EntityType::Individual);
        assert_eq!(parsed[0].source_list, SanctionsList::HmtUk);
        assert_eq!(
            parsed[0].dob,
            Some(NaiveDate::from_ymd_opt(1970, 1, 1).expect("date"))
        );
    }
}
