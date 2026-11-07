//! Deploy recipes keep the disk result tier writable by the
//! image's user (issue #396). The server probes its result
//! directory at boot and exits when it cannot write there,
//! so a recipe that mounts a root-owned volume for the
//! `nobody` image crash-loops instead of serving. Docker and
//! Kubernetes are not available to the test suite, so these
//! checks pin the recipe text that carries the fix: the
//! images create the directory owned by 65534, the compose
//! stack mounts its volume there (a fresh Docker named
//! volume copies that ownership), and the Kubernetes
//! recipes set the pod `fsGroup` that makes a mounted claim
//! group-writable by 65534.

use std::path::Path;

const RESULT_DIR: &str = "/var/cache/crabcrop";

fn read(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// The instructions of a Dockerfile's last stage, comments
/// dropped and line continuations joined, so a multi-line
/// `RUN` reads as one string.
fn final_stage(dockerfile: &str) -> Vec<String> {
    let mut instructions = Vec::new();
    let mut current = String::new();
    for line in dockerfile.lines() {
        let line = line.trim();
        if line.starts_with('#') || (line.is_empty() && current.is_empty()) {
            continue;
        }
        match line.strip_suffix('\\') {
            Some(head) => {
                current.push_str(head.trim_end());
                current.push(' ');
            }
            None => {
                current.push_str(line);
                instructions.push(std::mem::take(&mut current));
            }
        }
    }
    let last_from = instructions
        .iter()
        .rposition(|i| i.starts_with("FROM "))
        .expect("a FROM instruction");
    instructions.split_off(last_from)
}

/// The lines nested under the first `key:` that sits at
/// exactly `indent` spaces, up to the next line at that
/// indent or shallower. Enough YAML for these recipes.
fn yaml_block<'a>(text: &'a str, indent: usize, key: &str) -> Option<Vec<&'a str>> {
    let depth = |l: &str| l.len() - l.trim_start().len();
    let head = format!("{key}:");
    let mut lines = text.lines();
    lines.find(|l| depth(l) == indent && l.trim_start().starts_with(&head))?;
    Some(
        lines
            .take_while(|l| l.trim().is_empty() || depth(l) > indent)
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .collect(),
    )
}

#[test]
fn images_create_the_result_directory_for_nobody() {
    for rel in ["Dockerfile", "Dockerfile.release"] {
        let stage = final_stage(&read(rel));
        let user = stage
            .iter()
            .rposition(|i| i.starts_with("USER "))
            .unwrap_or_else(|| panic!("{rel}: the runtime stage sets a USER"));
        assert_eq!(stage[user], "USER nobody", "{rel}: runs as nobody");
        let prepares = stage[..user].iter().any(|i| {
            i.starts_with("RUN ")
                && i.contains(&format!("mkdir -p {RESULT_DIR}"))
                && i.contains(&format!("chown 65534:65534 {RESULT_DIR}"))
        });
        assert!(
            prepares,
            "{rel}: the runtime stage must create {RESULT_DIR} owned by 65534 \
             before USER nobody, or a fresh volume mounted there is root-owned \
             and the server exits at boot"
        );
    }
}

#[test]
fn compose_mounts_the_results_volume_at_the_image_directory() {
    // Docker seeds a fresh named volume from the image's
    // directory at the mount path, ownership included, so
    // the production stack is writable only while it mounts
    // `results` exactly where the image creates the directory.
    let compose = read("deploy/compose/docker-compose.prod.yml");
    assert!(
        compose.contains(&format!("CRABCROP_RESULT_STORAGE_DIR: {RESULT_DIR}")),
        "compose: the tier points at {RESULT_DIR}"
    );
    assert!(
        compose.contains(&format!("- results:{RESULT_DIR}\n")),
        "compose: the named volume is mounted at {RESULT_DIR}, the directory \
         the image creates owned by nobody"
    );
}

#[test]
fn kubernetes_recipes_set_the_pod_fs_group_for_nobody() {
    // Kustomize base: the pod-level securityContext is a
    // sibling of `containers:` under the pod template spec.
    // A container-level securityContext cannot carry fsGroup.
    let base = read("deploy/kustomize/base/deployment.yaml");
    let containers = base
        .lines()
        .find(|l| l.trim_start().starts_with("containers:"))
        .expect("containers in the base deployment");
    let pod_indent = containers.len() - containers.trim_start().len();
    let pod_security = yaml_block(&base, pod_indent, "securityContext")
        .expect("kustomize base: a pod-level securityContext");
    assert!(
        pod_security.iter().any(|l| l.trim() == "fsGroup: 65534"),
        "kustomize base: the pod securityContext sets fsGroup: 65534, or the \
         result-storage claim mounts root-owned and the server exits at boot"
    );

    // The full overlay composes result-storage onto that base.
    let full = read("deploy/kustomize/overlays/full/kustomization.yaml");
    assert!(full.contains("- ../../base"));
    assert!(full.contains("- ../../components/result-storage"));

    // Helm: the same setting, in the chart's default values.
    let values = read("deploy/helm/crabcrop/values.yaml");
    let pod_security = yaml_block(&values, 0, "podSecurityContext").expect("podSecurityContext");
    assert!(
        pod_security.iter().any(|l| l.trim() == "fsGroup: 65534"),
        "helm: podSecurityContext sets fsGroup: 65534"
    );
}

#[test]
fn the_helpers_read_what_they_claim() {
    let dockerfile = "FROM a AS b\nRUN x\nFROM c\n# note\nRUN one \\\n    && two\nUSER nobody\n";
    assert_eq!(
        final_stage(dockerfile),
        vec!["FROM c", "RUN one && two", "USER nobody"]
    );
    let yaml = "a:\n  b: 1\n  # c\n\n  d: 2\ne:\n  f: 3\n";
    assert_eq!(yaml_block(yaml, 0, "a").unwrap(), vec!["  b: 1", "  d: 2"]);
    assert_eq!(yaml_block(yaml, 0, "e").unwrap(), vec!["  f: 3"]);
    assert!(yaml_block(yaml, 2, "a").is_none());
}
