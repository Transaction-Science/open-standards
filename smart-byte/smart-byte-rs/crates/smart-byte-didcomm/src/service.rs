//! DID-document `service` block parser for `DIDCommMessaging` endpoints.
//!
//! Per DIDComm v2.1 § 7.4, an agent's DID document declares one or more
//! services of type `DIDCommMessaging` whose `serviceEndpoint` is either
//! a URI string, an object, or an array of objects with `uri`,
//! `accept`, and `routingKeys` members.

use serde::{Deserialize, Serialize};
use smart_byte_did::{Service, ServiceEndpoint};

use crate::error::DidcommError;

/// Service type discriminator used in DID documents.
pub const SERVICE_TYPE: &str = "DIDCommMessaging";

/// A parsed DIDComm service endpoint entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DidcommServiceEndpoint {
    /// HTTPS / WSS URI for delivery.
    pub uri: String,
    /// Accepted profiles (e.g. `didcomm/v2`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accept: Vec<String>,
    /// Routing keys (typically did:key URIs).
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "routingKeys")]
    pub routing_keys: Vec<String>,
}

/// Parse the `serviceEndpoint` of a `DIDCommMessaging` service into the
/// concrete endpoint list it represents.
pub fn parse_didcomm_service(
    service: &Service,
) -> Result<Vec<DidcommServiceEndpoint>, DidcommError> {
    if service.type_ != SERVICE_TYPE {
        return Err(DidcommError::Protocol(format!(
            "service type is not DIDCommMessaging: {}",
            service.type_
        )));
    }
    match &service.service_endpoint {
        ServiceEndpoint::Uri(uri) => Ok(vec![DidcommServiceEndpoint {
            uri: uri.clone(),
            ..Default::default()
        }]),
        ServiceEndpoint::Set(uris) => Ok(uris
            .iter()
            .map(|u| DidcommServiceEndpoint {
                uri: u.clone(),
                ..Default::default()
            })
            .collect()),
        ServiceEndpoint::Map(map) => {
            // Two shapes are common: a single endpoint object, or a
            // wrapper with a list under a known key.
            let v = serde_json::Value::Object(map.clone());
            if let Ok(single) = serde_json::from_value::<DidcommServiceEndpoint>(
                v.clone(),
            ) {
                return Ok(vec![single]);
            }
            // Otherwise: look for an array under "uri" or "endpoints".
            if let Some(arr) = v.get("endpoints").and_then(|x| x.as_array()) {
                let list: Vec<DidcommServiceEndpoint> = arr
                    .iter()
                    .filter_map(|x| {
                        serde_json::from_value::<DidcommServiceEndpoint>(x.clone())
                            .ok()
                    })
                    .collect();
                if list.is_empty() {
                    return Err(DidcommError::Protocol(
                        "endpoints array empty or malformed".into(),
                    ));
                }
                return Ok(list);
            }
            Err(DidcommError::Protocol(
                "unrecognised DIDCommMessaging serviceEndpoint object".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smart_byte_did::Service;

    #[test]
    fn parse_uri_endpoint() {
        let svc = Service {
            id: "#dc".into(),
            type_: "DIDCommMessaging".into(),
            service_endpoint: ServiceEndpoint::Uri(
                "https://example.com/dc".into(),
            ),
        };
        let parsed = parse_didcomm_service(&svc).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].uri, "https://example.com/dc");
    }

    #[test]
    fn parse_object_endpoint() {
        let mut m = serde_json::Map::new();
        m.insert("uri".into(), "https://example.com/dc".into());
        m.insert(
            "accept".into(),
            serde_json::Value::Array(vec!["didcomm/v2".into()]),
        );
        m.insert(
            "routingKeys".into(),
            serde_json::Value::Array(vec!["did:key:z6Mk...".into()]),
        );
        let svc = Service {
            id: "#dc".into(),
            type_: "DIDCommMessaging".into(),
            service_endpoint: ServiceEndpoint::Map(m),
        };
        let parsed = parse_didcomm_service(&svc).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].accept, vec!["didcomm/v2"]);
        assert_eq!(parsed[0].routing_keys, vec!["did:key:z6Mk..."]);
    }
}
