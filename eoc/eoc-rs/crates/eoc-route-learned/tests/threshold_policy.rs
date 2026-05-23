//! Threshold-policy behavioural tests.

use eoc_core::Stage;
use eoc_route_learned::router::StagePrediction;
use eoc_route_learned::threshold::{ThresholdDecision, ThresholdPolicy};

#[test]
fn confidence_below_threshold_does_not_skip() {
    let p = StagePrediction::new(Stage::Neural, 0.5);
    let policy = ThresholdPolicy::new(0.9);
    assert_eq!(policy.decide(&p), ThresholdDecision::FullCascade);
}

#[test]
fn confidence_above_threshold_skips() {
    let p = StagePrediction::new(Stage::Neural, 0.95);
    let policy = ThresholdPolicy::new(0.9);
    assert_eq!(policy.decide(&p), ThresholdDecision::SkipTo(Stage::Neural));
}

#[test]
fn at_threshold_skips() {
    let p = StagePrediction::new(Stage::Graph, 0.9);
    let policy = ThresholdPolicy::new(0.9);
    assert_eq!(policy.decide(&p), ThresholdDecision::SkipTo(Stage::Graph));
}
