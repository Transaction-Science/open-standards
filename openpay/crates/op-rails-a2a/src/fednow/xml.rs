//! `FedNow` pacs.008.001.08 emitter and pacs.002.001.10 parser.
//!
//! The full upstream `open-payments-iso20022` crate models the entire
//! schema, but instantiating the typed `Document` graph for every
//! emission has heavy serde overhead and tight coupling to the
//! upstream crate's internal naming. For `FedNow`'s narrow conformance
//! profile we emit the XML directly from a [`BuiltCreditTransfer<FedNow>`].
//!
//! The emitted XML matches the `FedNow` `MyStandards` profile (pacs.008.001.08)
//! and is what the FRB MQ endpoints accept on the credit-transfer queue.
//!
//! ## Parser scope
//!
//! For pacs.002 we extract exactly five fields:
//!
//! - `OrgnlUETR` — used to match request/response
//! - `OrgnlEndToEndId` — used as `rail_txn_id`
//! - `TxSts` — the transaction status code (ACSC, ACTC, RJCT, PDNG)
//! - `StsRsnInf/Rsn/Cd` — reason code on rejection
//! - `StsRsnInf/AddtlInf` — additional text
//!
//! We use a lightweight tag-scan parser instead of pulling in serde-xml
//! at this layer. This keeps the dependency graph small and guarantees
//! we don't break on schema-irrelevant whitespace changes.

use op_iso20022::BuiltCreditTransfer;
use op_iso20022::profile::FedNow;

// Re-export the shared helpers under their FedNow names so existing
// callers and tests keep working without touching the call sites.
pub use crate::xml_common::{
    ParsedPacs002, extract_first_tag, format_money, parse_pacs002, xml_escape,
};

/// Extract the ABA member id from a [`PartyIdentification::AbaRoutingNumber`].
fn aba_member(p: &op_iso20022::bah::PartyIdentification) -> &str {
    match p {
        op_iso20022::bah::PartyIdentification::AbaRoutingNumber(s)
        | op_iso20022::bah::PartyIdentification::Bic(s)
        | op_iso20022::bah::PartyIdentification::ClearingSystemMemberId { member_id: s, .. } => {
            s.as_str()
        }
    }
}

/// Emit a FedNow-profile pacs.008.001.08 XML document.
///
/// The output is the **business message body only** — without a BAH wrapper.
/// In `FedNow` the BAH (head.001) travels in MQ headers, not in the XML
/// body, so this is what we put in the MQ payload.
#[must_use]
pub fn emit_pacs008(
    built: &BuiltCreditTransfer<FedNow>,
    debtor_name: &str,
    creditor_name: &str,
    debtor_account: &str,
    creditor_account: &str,
) -> String {
    let amount = format_money(built.amount);
    let ccy = built.amount.currency.code();
    let debtor_rtn = aba_member(&built.bah.from);
    let creditor_rtn = aba_member(&built.bah.to);
    let creation_dt = built
        .bah
        .creation_datetime
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "2026-01-01T00:00:00Z".into());

    let remittance = built
        .remittance_info
        .as_deref()
        .map(|r| format!("<RmtInf><Ustrd>{}</Ustrd></RmtInf>", xml_escape(r)))
        .unwrap_or_default();

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08">
  <FIToFICstmrCdtTrf>
    <GrpHdr>
      <MsgId>{msg_id}</MsgId>
      <CreDtTm>{creation_dt}</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
      <SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf>
    </GrpHdr>
    <CdtTrfTxInf>
      <PmtId>
        <InstrId>{end_to_end_id}</InstrId>
        <EndToEndId>{end_to_end_id}</EndToEndId>
        <UETR>{uetr}</UETR>
      </PmtId>
      <IntrBkSttlmAmt Ccy="{ccy}">{amount}</IntrBkSttlmAmt>
      <ChrgBr>SLEV</ChrgBr>
      <InstgAgt><FinInstnId><ClrSysMmbId>
        <ClrSysId><Cd>USABA</Cd></ClrSysId>
        <MmbId>{debtor_rtn}</MmbId>
      </ClrSysMmbId></FinInstnId></InstgAgt>
      <InstdAgt><FinInstnId><ClrSysMmbId>
        <ClrSysId><Cd>USABA</Cd></ClrSysId>
        <MmbId>{creditor_rtn}</MmbId>
      </ClrSysMmbId></FinInstnId></InstdAgt>
      <Dbtr><Nm>{debtor_name_esc}</Nm></Dbtr>
      <DbtrAcct><Id><Othr><Id>{debtor_acct}</Id></Othr></Id></DbtrAcct>
      <DbtrAgt><FinInstnId><ClrSysMmbId>
        <ClrSysId><Cd>USABA</Cd></ClrSysId>
        <MmbId>{debtor_rtn}</MmbId>
      </ClrSysMmbId></FinInstnId></DbtrAgt>
      <CdtrAgt><FinInstnId><ClrSysMmbId>
        <ClrSysId><Cd>USABA</Cd></ClrSysId>
        <MmbId>{creditor_rtn}</MmbId>
      </ClrSysMmbId></FinInstnId></CdtrAgt>
      <Cdtr><Nm>{creditor_name_esc}</Nm></Cdtr>
      <CdtrAcct><Id><Othr><Id>{creditor_acct}</Id></Othr></Id></CdtrAcct>
      {remittance}
    </CdtTrfTxInf>
  </FIToFICstmrCdtTrf>
</Document>"#,
        msg_id = built.bah.business_message_id,
        creation_dt = creation_dt,
        end_to_end_id = built.end_to_end_id,
        uetr = built.uetr,
        ccy = ccy,
        amount = amount,
        debtor_rtn = debtor_rtn,
        creditor_rtn = creditor_rtn,
        debtor_name_esc = xml_escape(debtor_name),
        creditor_name_esc = xml_escape(creditor_name),
        debtor_acct = xml_escape(debtor_account),
        creditor_acct = xml_escape(creditor_account),
        remittance = remittance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{A2aKey, Currency, Money, PaymentMethod};
    use op_iso20022::CreditTransferBuilder;
    use op_iso20022::bah::PartyIdentification;

    fn built_transfer() -> BuiltCreditTransfer<FedNow> {
        CreditTransferBuilder::<FedNow>::new()
            .amount(Money::from_minor(12345, Currency::USD))
            .debtor(PaymentMethod::A2a(A2aKey::UsAch {
                routing: "021000021".into(),
                account: "1234567890".into(),
            }))
            .creditor(PaymentMethod::A2a(A2aKey::UsAch {
                routing: "026009593".into(),
                account: "9876543210".into(),
            }))
            .debtor_agent(PartyIdentification::AbaRoutingNumber("021000021".into()))
            .creditor_agent(PartyIdentification::AbaRoutingNumber("026009593".into()))
            .end_to_end_id("e2e_001")
            .uetr("12a345b6-7c89-4d01-23e4-567890abcdef")
            .remittance("Invoice 4242")
            .build()
            .expect("builder should succeed")
    }

    #[test]
    fn emit_pacs008_contains_required_fields() {
        let built = built_transfer();
        let xml = emit_pacs008(
            &built,
            "Alice Sender",
            "Bob Recipient",
            "1234567890",
            "9876543210",
        );
        assert!(xml.contains(r#"xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08""#));
        assert!(xml.contains("<UETR>12a345b6-7c89-4d01-23e4-567890abcdef</UETR>"));
        assert!(xml.contains("<EndToEndId>e2e_001</EndToEndId>"));
        assert!(xml.contains(r#"<IntrBkSttlmAmt Ccy="USD">123.45</IntrBkSttlmAmt>"#));
        assert!(xml.contains("<SttlmMtd>CLRG</SttlmMtd>"));
        assert!(xml.contains("<ChrgBr>SLEV</ChrgBr>"));
        assert!(xml.contains("Alice Sender"));
        assert!(xml.contains("Bob Recipient"));
        assert!(xml.contains("<Ustrd>Invoice 4242</Ustrd>"));
        assert!(xml.contains("<MmbId>021000021</MmbId>"));
        assert!(xml.contains("<MmbId>026009593</MmbId>"));
    }

    #[test]
    fn emit_pacs008_escapes_xml_special_chars() {
        let built = built_transfer();
        let xml = emit_pacs008(
            &built,
            "Tom & Jerry <Co>",
            "AT&T",
            "1234567890",
            "9876543210",
        );
        assert!(xml.contains("Tom &amp; Jerry &lt;Co&gt;"));
        assert!(xml.contains("AT&amp;T"));
        // Original chars must NOT appear in the output.
        assert!(!xml.contains("Tom & Jerry"));
    }

    #[test]
    fn emit_pacs008_omits_remittance_block_when_none() {
        let built = CreditTransferBuilder::<FedNow>::new()
            .amount(Money::from_minor(100, Currency::USD))
            .debtor(PaymentMethod::A2a(A2aKey::UsAch {
                routing: "021000021".into(),
                account: "111".into(),
            }))
            .creditor(PaymentMethod::A2a(A2aKey::UsAch {
                routing: "026009593".into(),
                account: "222".into(),
            }))
            .debtor_agent(PartyIdentification::AbaRoutingNumber("021000021".into()))
            .creditor_agent(PartyIdentification::AbaRoutingNumber("026009593".into()))
            .end_to_end_id("e2e_noremit")
            .uetr("12a345b6-7c89-4d01-23e4-567890abcd00")
            .build()
            .unwrap();
        let xml = emit_pacs008(&built, "A", "B", "111", "222");
        assert!(!xml.contains("RmtInf"));
        assert!(!xml.contains("Ustrd"));
    }
}
