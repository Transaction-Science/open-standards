//! Prometheus text exposition format.

use eoc_observability::{Counter, Gauge, Histogram, PrometheusExposer};

#[test]
fn counter_emits_help_type_and_value() {
    let c = Counter::new("eoc_cache_hits_total", "1")
        .with_description("Total cache hits")
        .with_attribute("stage", "cache");
    c.add(7);

    let mut e = PrometheusExposer::new();
    e.add_counter(&c);
    let out = e.finish();

    assert!(out.contains("# HELP eoc_cache_hits_total Total cache hits"));
    assert!(out.contains("# TYPE eoc_cache_hits_total counter"));
    assert!(out.contains("eoc_cache_hits_total{stage=\"cache\"} 7"));
}

#[test]
fn gauge_renders_no_attrs() {
    let g = Gauge::new("eoc_active_connections", "1");
    g.set(3.5);
    let mut e = PrometheusExposer::new();
    e.add_gauge(&g);
    let out = e.finish();
    assert!(out.contains("# TYPE eoc_active_connections gauge"));
    assert!(out.contains("eoc_active_connections 3.5"));
}

#[test]
fn histogram_emits_cumulative_buckets() {
    let h = Histogram::new(
        "eoc_latency_ms",
        "ms",
        vec![10.0, 50.0, 100.0, 500.0],
    );
    h.record(5.0); // bucket 0 (<=10)
    h.record(40.0); // bucket 1 (<=50)
    h.record(75.0); // bucket 2 (<=100)
    h.record(200.0); // bucket 3 (<=500)
    h.record(2000.0); // bucket 4 (+Inf)

    let mut e = PrometheusExposer::new();
    e.add_histogram(&h);
    let out = e.finish();

    assert!(out.contains("# TYPE eoc_latency_ms histogram"));
    assert!(out.contains("eoc_latency_ms_bucket{le=\"10\"} 1"));
    assert!(out.contains("eoc_latency_ms_bucket{le=\"50\"} 2"));
    assert!(out.contains("eoc_latency_ms_bucket{le=\"100\"} 3"));
    assert!(out.contains("eoc_latency_ms_bucket{le=\"500\"} 4"));
    assert!(out.contains("eoc_latency_ms_bucket{le=\"+Inf\"} 5"));
    assert!(out.contains("eoc_latency_ms_count 5"));
    assert!(out.contains("eoc_latency_ms_sum 2320"));
}
