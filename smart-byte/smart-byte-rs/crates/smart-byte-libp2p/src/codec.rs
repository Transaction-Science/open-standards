//! CBOR codec for point-to-point envelope fetch via `request_response`.
//!
//! A request is a [`Said`] (the SAID of the requested envelope).
//! A response is an optional [`Envelope`]; `None` means the responder
//! does not hold the envelope. We frame each side with a u32 length
//! prefix so the wire is self-delimiting.

use std::io;

use async_trait::async_trait;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::StreamProtocol;
use libp2p::request_response::Codec;
use serde::{Deserialize, Serialize};
use smart_byte_core::{Envelope, Said};

/// Wire-level request: ask for the envelope whose SAID is `said`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvelopeRequest {
    /// The SAID being requested.
    pub said: Said,
}

/// Wire-level response: either the requested envelope or `None`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvelopeResponse {
    /// The envelope if the responder holds it, otherwise `None`.
    pub envelope: Option<Envelope>,
}

/// libp2p `Codec` implementation for [`EnvelopeRequest`] /
/// [`EnvelopeResponse`].
#[derive(Default, Clone)]
pub struct EnvelopeCodec;

/// Maximum frame size on the request-response wire. 16 MiB is enough
/// headroom for any envelope cargo we expect today while still keeping
/// nodes safe against malicious peers.
const MAX_FRAME: usize = 16 * 1024 * 1024;

async fn read_frame<R>(io: &mut R) -> io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin + Send,
{
    let mut len_bytes = [0u8; 4];
    io.read_exact(&mut len_bytes).await?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame {len} bytes exceeds max {MAX_FRAME}"),
        ));
    }
    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn write_frame<W>(io: &mut W, bytes: &[u8]) -> io::Result<()>
where
    W: AsyncWrite + Unpin + Send,
{
    if bytes.len() > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame {} bytes exceeds max {MAX_FRAME}", bytes.len()),
        ));
    }
    let len = (bytes.len() as u32).to_be_bytes();
    io.write_all(&len).await?;
    io.write_all(bytes).await?;
    io.flush().await?;
    Ok(())
}

#[async_trait]
impl Codec for EnvelopeCodec {
    type Protocol = StreamProtocol;
    type Request = EnvelopeRequest;
    type Response = EnvelopeResponse;

    async fn read_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_frame(io).await?;
        serde_cbor::from_slice(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    async fn read_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_frame(io).await?;
        serde_cbor::from_slice(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let bytes = serde_cbor::to_vec(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        write_frame(io, &bytes).await
    }

    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let bytes = serde_cbor::to_vec(&resp)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        write_frame(io, &bytes).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use futures::io::Cursor;
    use smart_byte_core::{Cargo, JouleCost, OwnershipChain, Provenance};

    fn fixture_envelope() -> Envelope {
        let issuer = Said::hash(b"issuer-codec");
        let issued_at = chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap();
        let prov = Provenance::new(issuer, issued_at, vec![]);
        Envelope::new(
            prov,
            OwnershipChain::empty(),
            Cargo::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            JouleCost::measured(7),
        )
        .expect("envelope")
    }

    #[tokio::test]
    async fn request_roundtrip() {
        let proto = StreamProtocol::new("/smart-byte/envelope/1.0.0");
        let mut codec = EnvelopeCodec;
        let env = fixture_envelope();
        let req = EnvelopeRequest { said: env.id };

        let mut buf = Vec::new();
        {
            let mut w = Cursor::new(&mut buf);
            codec
                .write_request(&proto, &mut w, req.clone())
                .await
                .expect("write");
        }
        let mut r = Cursor::new(&buf[..]);
        let got = codec.read_request(&proto, &mut r).await.expect("read");
        assert_eq!(got, req);
    }

    #[tokio::test]
    async fn response_roundtrip() {
        let proto = StreamProtocol::new("/smart-byte/envelope/1.0.0");
        let mut codec = EnvelopeCodec;
        let env = fixture_envelope();
        let resp = EnvelopeResponse {
            envelope: Some(env.clone()),
        };

        let mut buf = Vec::new();
        {
            let mut w = Cursor::new(&mut buf);
            codec
                .write_response(&proto, &mut w, resp.clone())
                .await
                .expect("write");
        }
        let mut r = Cursor::new(&buf[..]);
        let got = codec.read_response(&proto, &mut r).await.expect("read");
        assert_eq!(got, resp);
        let back = got.envelope.expect("envelope");
        back.verify_said().expect("said still valid");
    }

    #[tokio::test]
    async fn response_none_roundtrip() {
        let proto = StreamProtocol::new("/smart-byte/envelope/1.0.0");
        let mut codec = EnvelopeCodec;
        let resp = EnvelopeResponse { envelope: None };

        let mut buf = Vec::new();
        {
            let mut w = Cursor::new(&mut buf);
            codec
                .write_response(&proto, &mut w, resp.clone())
                .await
                .expect("write");
        }
        let mut r = Cursor::new(&buf[..]);
        let got = codec.read_response(&proto, &mut r).await.expect("read");
        assert_eq!(got, resp);
    }
}
