//! `FedNow` MQ transport abstraction.
//!
//! Operators bridge `OpenPay` to their IBM MQ queue manager by implementing
//! [`MqChannel`]. The bridge can be:
//!
//! - A long-lived JMS client process in a sidecar
//! - A connection pool to the `FedLine` Direct MQ endpoint
//! - A local MQ proxy (FRB itself runs the queue manager; the
//!   participant runs the client)
//!
//! `OpenPay` does not embed an IBM MQ client. MQ is heavy, license-bound,
//! and operationally specific to each deployment. The [`MqChannel`]
//! trait is the seam.
//!
//! ## Message format
//!
//! Bodies are XML pacs.008 (`FedNow` profile = pacs.008.001.08, per
//! verified specs). Properties carry routing headers `FedNow`'s queue
//! manager expects (`MQRO_*`, `MQMD.Format`, BAH headers).

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// An outbound MQ message.
///
/// `payload` is the XML body. `properties` carry MQ message properties
/// `FedNow` uses for routing (queue name, correlation id, BAH metadata).
/// We pass them through unchanged so the bridge stays thin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqMessage {
    /// Logical queue name to send to (e.g. `FRB.FEDNOW.PACS008.IN`).
    pub queue: String,
    /// XML payload bytes (UTF-8).
    pub payload: Vec<u8>,
    /// Correlation id for matching responses. Use the UETR.
    pub correlation_id: String,
    /// Free-form headers / MQ properties forwarded to the queue manager.
    pub properties: Vec<(String, String)>,
}

/// An inbound MQ message returned by the bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqResponse {
    /// XML response body (typically a pacs.002).
    pub payload: Vec<u8>,
    /// Correlation id echoed by the rail.
    pub correlation_id: String,
    /// Optional negative ack code from MQ itself (e.g. dead-letter).
    pub mq_ack_code: Option<String>,
}

/// Synchronous request/response over an MQ transport. Implementations
/// are free to use connection pools, async runtimes internally, etc.
pub trait MqChannel: Send + Sync {
    /// Send `message`, await the matching response by correlation id,
    /// and return it. The bridge handles routing back to the right
    /// reply queue.
    fn request(&self, message: &MqMessage) -> Result<MqResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use std::sync::Mutex;

    /// Test double that records the request and returns a canned reply.
    struct CannedMq {
        last_request: Mutex<Option<MqMessage>>,
        canned_response: MqResponse,
    }

    impl MqChannel for CannedMq {
        fn request(&self, m: &MqMessage) -> Result<MqResponse> {
            *self.last_request.lock().unwrap() = Some(m.clone());
            Ok(self.canned_response.clone())
        }
    }

    #[test]
    fn mq_channel_round_trips() {
        let canned = MqResponse {
            payload: b"<Document>...</Document>".to_vec(),
            correlation_id: "uetr-1".into(),
            mq_ack_code: None,
        };
        let channel = CannedMq {
            last_request: Mutex::new(None),
            canned_response: canned.clone(),
        };
        let req = MqMessage {
            queue: "FRB.FEDNOW.PACS008.IN".into(),
            payload: b"<Document>pacs.008</Document>".to_vec(),
            correlation_id: "uetr-1".into(),
            properties: vec![("BAH.From".into(), "021000021".into())],
        };
        let resp = channel.request(&req).unwrap();
        assert_eq!(resp.correlation_id, "uetr-1");
        assert_eq!(resp.payload, canned.payload);
        let recorded = channel.last_request.lock().unwrap();
        assert!(recorded.is_some());
        assert_eq!(recorded.as_ref().unwrap().queue, "FRB.FEDNOW.PACS008.IN");
    }

    #[test]
    fn mq_channel_is_object_safe() {
        // Must be able to hold dyn MqChannel.
        struct Failing;
        impl MqChannel for Failing {
            fn request(&self, _: &MqMessage) -> Result<MqResponse> {
                Err(Error::Transport("disconnected".into()))
            }
        }
        let _: Box<dyn MqChannel> = Box::new(Failing);
    }
}
