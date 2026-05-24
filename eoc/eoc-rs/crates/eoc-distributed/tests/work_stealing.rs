//! Work-stealing scheduler balances load between heterogeneous workers.

use eoc_distributed::{
    Accelerator, Capability, InMemoryWorker, Load, WorkItem, WorkStealingScheduler, Worker,
};

fn cpu(id: &str, micro_j: u32) -> InMemoryWorker {
    InMemoryWorker::new(
        id,
        Capability {
            models: vec!["m".into()],
            accelerator: Accelerator::Cpu,
            max_concurrency: 8,
            continuous_batching: false,
            paged_kv: false,
            zone: "EU-FR".into(),
        },
        Load {
            micro_joules_per_token: micro_j,
            ..Load::idle()
        },
    )
}

fn gpu(id: &str, micro_j: u32) -> InMemoryWorker {
    InMemoryWorker::new(
        id,
        Capability {
            models: vec!["m".into()],
            accelerator: Accelerator::Gpu,
            max_concurrency: 16,
            continuous_batching: true,
            paged_kv: true,
            zone: "EU-FR".into(),
        },
        Load {
            micro_joules_per_token: micro_j,
            ..Load::idle()
        },
    )
}

#[test]
fn submit_lands_in_lowest_joule_worker() {
    let mut s = WorkStealingScheduler::new();
    let a = cpu("a", 200);
    let b = cpu("b", 50);
    let pool: Vec<&dyn Worker> = vec![&a, &b];
    s.ensure(&pool);

    for i in 0..5 {
        s.submit(
            &pool,
            WorkItem {
                id: format!("r{i}"),
                model: "m".into(),
                expected_tokens: 32,
                require: None,
                locality_hint: None,
            },
        )
        .expect("ok");
    }
    assert_eq!(s.depth("b"), 5);
    assert_eq!(s.depth("a"), 0);
}

#[test]
fn idle_worker_steals_from_busy() {
    let mut s = WorkStealingScheduler::new();
    let a = cpu("a", 100);
    let b = cpu("b", 100);
    let pool: Vec<&dyn Worker> = vec![&a, &b];
    s.ensure(&pool);

    // Pile four items directly into a's inbox via repeated submits
    // (both are CPU workers with equal joule cost — submit goes to
    // whichever min_by sees first; doesn't matter for this test).
    for i in 0..4 {
        s.submit(
            &pool,
            WorkItem {
                id: format!("r{i}"),
                model: "m".into(),
                expected_tokens: 16,
                require: None,
                locality_hint: None,
            },
        )
        .expect("ok");
    }
    // Find the worker that ended up loaded.
    let (loaded, empty) = if s.depth("a") > s.depth("b") {
        ("a", "b")
    } else {
        ("b", "a")
    };
    assert!(s.depth(loaded) >= 2);
    assert_eq!(s.depth(empty), 0);

    let stolen = s.steal(&pool, empty).expect("steal");
    assert!(!stolen.is_empty());
    assert_eq!(s.depth(empty), 1);
    assert_eq!(s.depth(loaded), 3);
}

#[test]
fn capability_filters_block_cross_class_steal() {
    let mut s = WorkStealingScheduler::new();
    let cpu_a = cpu("cpu-a", 100);
    let gpu_b = gpu("gpu-b", 100);
    let pool: Vec<&dyn Worker> = vec![&cpu_a, &gpu_b];
    s.ensure(&pool);

    // Submit a GPU-requiring item; it lands on the GPU worker.
    s.submit(
        &pool,
        WorkItem {
            id: "gpu-only".into(),
            model: "m".into(),
            expected_tokens: 32,
            require: Some(Accelerator::Gpu),
            locality_hint: None,
        },
    )
    .expect("ok");
    assert_eq!(s.depth("gpu-b"), 1);

    // CPU worker tries to steal — must fail (returns None).
    let stolen = s.steal(&pool, "cpu-a");
    assert!(stolen.is_none());
    assert_eq!(s.depth("gpu-b"), 1);
    assert_eq!(s.depth("cpu-a"), 0);
}
