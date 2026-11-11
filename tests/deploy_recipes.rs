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

/// The images' runtime user: nobody, by number. The kubelet
/// checks `runAsNonRoot` only against a numeric user.
const RUNTIME_USER: &str = "USER 65534:65534";

fn read(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// The instructions of a Dockerfile, comments dropped and
/// line continuations joined, so a multi-line `RUN` reads as
/// one string.
fn instructions(dockerfile: &str) -> Vec<String> {
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
    instructions
}

/// The instructions of a Dockerfile's last stage, as
/// [`instructions`] reads them.
fn final_stage(dockerfile: &str) -> Vec<String> {
    let mut instructions = instructions(dockerfile);
    let last_from = instructions
        .iter()
        .rposition(|i| i.starts_with("FROM "))
        .expect("a FROM instruction");
    instructions.split_off(last_from)
}

/// Every stage of a Dockerfile, in order, each from its
/// `FROM`. What comes before the first `FROM` is left out.
fn stages(dockerfile: &str) -> Vec<Vec<String>> {
    let mut stages: Vec<Vec<String>> = Vec::new();
    for instruction in instructions(dockerfile) {
        if instruction.starts_with("FROM ") {
            stages.push(vec![instruction]);
        } else if let Some(stage) = stages.last_mut() {
            stage.push(instruction);
        }
    }
    stages
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
        assert_eq!(stage[user], RUNTIME_USER, "{rel}: runs as nobody");
        let prepares = stage[..user].iter().any(|i| {
            i.starts_with("RUN ")
                && i.contains(&format!("mkdir -p {RESULT_DIR}"))
                && i.contains(&format!("chown 65534:65534 {RESULT_DIR}"))
        });
        assert!(
            prepares,
            "{rel}: the runtime stage must create {RESULT_DIR} owned by 65534 \
             before {RUNTIME_USER}, or a fresh volume mounted there is root-owned \
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
    let dockerfile =
        "ARG A=1\nFROM a AS b\nRUN x\nFROM c\n# note\nRUN one \\\n    && two\nUSER nobody\n";
    assert_eq!(
        final_stage(dockerfile),
        vec!["FROM c", "RUN one && two", "USER nobody"]
    );
    assert_eq!(
        stages(dockerfile),
        vec![
            vec!["FROM a AS b", "RUN x"],
            vec!["FROM c", "RUN one && two", "USER nobody"]
        ]
    );
    let yaml = "a:\n  b: 1\n  # c\n\n  d: 2\ne:\n  f: 3\n";
    assert_eq!(yaml_block(yaml, 0, "a").unwrap(), vec!["  b: 1", "  d: 2"]);
    assert_eq!(yaml_block(yaml, 0, "e").unwrap(), vec!["  f: 3"]);
    assert!(yaml_block(yaml, 2, "a").is_none());
}

/// The lines of a YAML text whose key is one of `keys`,
/// list-item dashes included and comments skipped.
fn yaml_keys<'a>(text: &'a str, keys: &[&str]) -> Vec<&'a str> {
    text.lines()
        .filter(|l| {
            let t = l.trim_start();
            let t = t.strip_prefix("- ").unwrap_or(t).trim_start();
            keys.iter().any(|k| t.starts_with(&format!("{k}:")))
        })
        .collect()
}

/// Every YAML file under `dir`, recursively, sorted.
fn yaml_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(yaml_files(&path));
        } else if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
            found.push(path);
        }
    }
    found.sort();
    found
}

#[test]
fn recipes_start_the_server_with_no_arguments() {
    // #470: any argument at all runs the CLI, and one it
    // does not know exits 1, so a recipe that handed the
    // server an argument would crash-loop instead of
    // serving. The images run a bare `crabcrop` (the AI
    // image inherits that entrypoint), and no recipe adds a
    // command, an entrypoint, or arguments.
    for rel in ["Dockerfile", "Dockerfile.release"] {
        let stage = final_stage(&read(rel));
        let entrypoints: Vec<&str> = stage
            .iter()
            .map(String::as_str)
            .filter(|i| i.starts_with("ENTRYPOINT"))
            .collect();
        assert_eq!(entrypoints, [r#"ENTRYPOINT ["crabcrop"]"#], "{rel}");
        assert!(
            !stage.iter().any(|i| i.starts_with("CMD")),
            "{rel}: a CMD would become the server's arguments"
        );
    }
    let ai = final_stage(&read("Dockerfile.ai"));
    assert_eq!(ai[0], "FROM ${BASE_IMAGE}");
    assert!(
        !ai.iter()
            .any(|i| i.starts_with("ENTRYPOINT") || i.starts_with("CMD")),
        "Dockerfile.ai: inherits the base image's bare entrypoint"
    );

    let unit = read("deploy/vm/crabcrop.service");
    let exec: Vec<&str> = unit
        .lines()
        .filter(|l| l.starts_with("ExecStart="))
        .collect();
    assert_eq!(exec, ["ExecStart=/usr/local/bin/crabcrop"]);

    // Compose spells a container's arguments `command:` and
    // `entrypoint:` (its `args:` are build arguments);
    // Kubernetes, Cloud Run and Container Apps spell them
    // `command:` and `args:`.
    let deploy = Path::new(env!("CARGO_MANIFEST_DIR")).join("deploy");
    let files = yaml_files(&deploy);
    assert!(files.len() >= 10, "found the YAML recipes: {files:?}");
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let compose = path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("docker-compose"));
        let keys: &[&str] = if compose {
            &["command", "entrypoint"]
        } else {
            &["command", "args"]
        };
        let found = yaml_keys(&text, keys);
        let rel = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap();
        assert!(found.is_empty(), "{}: {found:?}", rel.display());
    }

    let task: serde_json::Value =
        serde_json::from_str(&read("deploy/ecs-fargate/task-definition.json")).unwrap();
    let containers = task["containerDefinitions"].as_array().unwrap();
    assert!(!containers.is_empty());
    for container in containers {
        for key in ["command", "entryPoint"] {
            assert!(
                container.get(key).is_none(),
                "ecs-fargate: {key} in {container}"
            );
        }
    }

    // The YAML check is not vacuous.
    let yaml = "containers:\n  - name: c\n    args: [\"--x\"]\n    # command: y\n  - command:\n";
    assert_eq!(
        yaml_keys(yaml, &["command", "args"]),
        vec!["    args: [\"--x\"]", "  - command:"]
    );
}

#[test]
fn the_helm_chart_gives_the_pod_more_than_the_server_drain() {
    // Issue #463: the server drains for its shutdown grace
    // and then exits. The chart passes its grace to the
    // server and sets the pod's termination grace from the
    // same value plus a margin, so the kubelet cannot kill
    // the pod mid-drain; the default matches the server's.
    let values = read("deploy/helm/crabcrop/values.yaml");
    let grace: Vec<&str> = values
        .lines()
        .filter_map(|l| l.strip_prefix("shutdownGraceSeconds:"))
        .map(str::trim)
        .collect();
    assert_eq!(
        grace,
        [crabcrop::server::DEFAULT_SHUTDOWN_GRACE_SECS.to_string()],
        "values.yaml: shutdownGraceSeconds is the server's default grace"
    );

    let deployment = read("deploy/helm/crabcrop/templates/deployment.yaml");
    let pod_grace: Vec<&str> = deployment
        .lines()
        .filter_map(|l| l.trim().strip_prefix("terminationGracePeriodSeconds:"))
        .map(str::trim)
        .collect();
    let [pod_grace] = pod_grace[..] else {
        panic!("deployment.yaml: one terminationGracePeriodSeconds, got {pod_grace:?}");
    };
    let margin = pod_grace
        .strip_prefix("{{ add .Values.shutdownGraceSeconds ")
        .and_then(|rest| rest.strip_suffix(" }}"))
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("deployment.yaml: {pod_grace:?} is not the grace plus a margin"));
    assert!(margin > 0, "{pod_grace}");

    let env = deployment
        .lines()
        .position(|l| l.trim() == "- name: CRABCROP_SHUTDOWN_GRACE_SECS")
        .expect("deployment.yaml: the server gets CRABCROP_SHUTDOWN_GRACE_SECS");
    assert_eq!(
        deployment.lines().nth(env + 1).map(str::trim),
        Some("value: {{ .Values.shutdownGraceSeconds | quote }}"),
        "deployment.yaml: CRABCROP_SHUTDOWN_GRACE_SECS comes from shutdownGraceSeconds"
    );

    // The same name in `env` or `ai.env` would render a
    // second entry, which wins and leaves the pod's grace
    // short of the server's: the chart refuses it.
    let guard = deployment
        .lines()
        .position(|l| l.trim() == "{{- if hasKey $allEnv \"CRABCROP_SHUTDOWN_GRACE_SECS\" }}")
        .expect("deployment.yaml: refuses CRABCROP_SHUTDOWN_GRACE_SECS in env");
    assert!(
        deployment
            .lines()
            .nth(guard + 1)
            .is_some_and(|l| l.trim().starts_with("{{- fail ")),
        "deployment.yaml: the guard fails the render"
    );
    let merged = deployment
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("set $allEnv $k $v"))
        .map(|(i, _)| i)
        .last()
        .expect("deployment.yaml: env and ai.env merge into $allEnv");
    let listed = deployment
        .lines()
        .position(|l| l.contains("range $k := keys $allEnv"))
        .expect("deployment.yaml: $allEnv is listed");
    assert!(
        merged < guard && guard < listed,
        "deployment.yaml: the guard checks the merged env before it is listed"
    );
}

/// The client hint headers, read from the `Vary` row of
/// docs/responses.md (which a server unit test ties to the
/// header the server sends): every header named there but
/// `Accept`.
fn hint_headers() -> Vec<String> {
    let responses = read("docs/responses.md");
    let row = responses
        .lines()
        .find(|l| l.starts_with("| `Vary` |"))
        .expect("docs/responses.md: a Vary row");
    let value = row.split('`').nth(3).expect("the Vary row's first value");
    let names: Vec<String> = value
        .split(", ")
        .filter(|n| *n != "Accept")
        .map(str::to_string)
        .collect();
    assert_eq!(names.len(), 7, "{value}");
    names
}

/// The CDN recipes whose edge keys on neither `Vary` nor
/// the client hint headers.
const UNKEYED_HINT_RECIPES: [&str; 4] = [
    "deploy/cloudflare/README.md",
    "deploy/cloudfront/README.md",
    "deploy/bunnycdn/README.md",
    "deploy/akamai/README.md",
];

#[test]
fn recipes_whose_edge_cannot_key_on_hints_turn_them_off() {
    // A hint header changes a URL's bytes, and these edges
    // store one copy per key: with hints read at the
    // origin, whichever request filled the edge would pick
    // the copy every later viewer gets. Each recipe turns
    // hints off on the origin.
    for rel in UNKEYED_HINT_RECIPES {
        assert!(
            read(rel).contains("`CRABCROP_CLIENT_HINTS=0`"),
            "{rel}: sets CRABCROP_CLIENT_HINTS=0 on the origin"
        );
    }
}

#[test]
fn the_cloudfront_origin_request_policy_forwards_no_hint() {
    let recipe = read("deploy/cloudfront/README.md");
    let forwarded = recipe
        .lines()
        .find(|l| l.starts_with("- Headers:"))
        .expect("cloudfront: the origin request policy's headers");
    assert!(forwarded.contains("`Accept`"), "{forwarded}");
    for hint in hint_headers() {
        assert!(
            !forwarded.contains(&format!("`{hint}`")),
            "cloudfront: forwards {hint}, which no key carries: {forwarded}"
        );
    }
}

#[test]
fn the_akamai_cache_key_carries_every_hint_header_or_none() {
    // Remove Vary Header leaves the key as the only thing
    // that tells variants apart: a hint the key leaves out
    // would share one object between the bytes it changes.
    let rules: serde_json::Value =
        serde_json::from_str(&read("deploy/akamai/property-rules.json")).unwrap();
    let cache_id = rules["rules"]["behaviors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["name"] == "cacheId")
        .expect("akamai: a cacheId behavior");
    let keyed: Vec<&str> = cache_id["options"]["elements"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_str().unwrap())
        .collect();
    assert!(keyed.contains(&"Accept"), "{keyed:?}");
    let hints = hint_headers();
    let keyed_hints = hints.iter().filter(|h| keyed.contains(&h.as_str())).count();
    assert!(
        keyed_hints == 0 || keyed_hints == hints.len(),
        "akamai: keys {keyed_hints} of the {} hint headers: {keyed:?}",
        hints.len()
    );
}

#[test]
fn the_cloudfront_recipe_keeps_errors_out_of_its_error_cache() {
    // CloudFront caches an error on its own for the error
    // caching minimum TTL, whatever the origin's headers:
    // every error status the image route answers gets a
    // custom error response with that TTL at 0.
    let recipe = read("deploy/cloudfront/README.md");
    let setting = recipe
        .lines()
        .find(|l| l.contains("**Error caching minimum TTL: 0**"))
        .expect("cloudfront: custom error responses with an error caching TTL of 0");
    for status in [400, 403, 404, 405, 500, 501, 502, 503, 504] {
        assert!(
            setting.contains(&format!("`{status}`")),
            "{status}: {setting}"
        );
    }
}

#[test]
fn cdn_recipes_name_the_error_header_the_server_sends() {
    // Errors leave with `Cache-Control: no-store` (a server
    // test pins it); a recipe that tells operators anything
    // else about them is wrong about what reaches the edge.
    for rel in [
        "deploy/cloudflare/README.md",
        "deploy/cloudfront/README.md",
        "deploy/fastly/README.md",
        "deploy/bunnycdn/README.md",
        "deploy/akamai/README.md",
    ] {
        let recipe = read(rel);
        assert!(
            recipe.contains("error responses (`no-store`)")
                || recipe.contains("error responses `no-store`"),
            "{rel}: names the error responses' no-store"
        );
        assert!(
            !recipe.contains("error responses `private, no-store`"),
            "{rel}: error responses are no-store, not private, no-store"
        );
    }
}

/// The variables the server treats as secrets: the names in
/// `SECRET_VARIABLES` in src/config.rs. That module belongs
/// to the binary, which an integration test cannot import,
/// so the list is read from the source.
fn server_secret_variables() -> Vec<String> {
    let config = read("src/config.rs");
    let head = "const SECRET_VARIABLES: [&str; ";
    let start = config.find(head).expect("src/config.rs: SECRET_VARIABLES");
    let rest = &config[start + head.len()..];
    let len: usize = rest[..rest.find(']').unwrap()].parse().unwrap();
    let body = &rest[rest.find("= [").unwrap()..rest.find("];").unwrap()];
    let names: Vec<String> = body
        .split('"')
        .skip(1)
        .step_by(2)
        .map(String::from)
        .collect();
    assert_eq!(names.len(), len, "SECRET_VARIABLES: {names:?}");
    names
}

/// Key IDs the server reads next to a secret: not secret
/// themselves, but set together with it, so the chart reads
/// each from the same Secret as its partner.
const SECRET_PARTNERS: [(&str, &str); 2] = [
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    ("CRABCROP_GCS_ACCESS_KEY", "CRABCROP_GCS_SECRET_KEY"),
];

/// The env entries of a chart template that read a Secret
/// (`- name:` then `valueFrom:` then `secretKeyRef:`), each
/// with the closest `{{- if ... }}` above it, the condition
/// that renders it.
fn secret_ref_entries(template: &str) -> std::collections::BTreeMap<String, String> {
    let lines: Vec<&str> = template.lines().map(str::trim).collect();
    let mut found = std::collections::BTreeMap::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(name) = line.strip_prefix("- name: ") else {
            continue;
        };
        if lines.get(i + 1..i + 3) != Some(&["valueFrom:", "secretKeyRef:"][..]) {
            continue;
        }
        let guard = lines[..i]
            .iter()
            .rev()
            .find(|l| l.starts_with("{{- if "))
            .map_or(String::new(), |l| l.to_string());
        found.insert(name.to_string(), guard);
    }
    found
}

/// The lines of a named template, `{{- define "<name>" -}}`
/// up to the `{{- end }}` that closes it, with each action
/// on its own line as the chart writes them.
fn template_define<'a>(text: &'a str, name: &str) -> Option<Vec<&'a str>> {
    let open = format!("{{{{- define \"{name}\" -}}}}");
    let mut lines = text.lines();
    lines.find(|l| l.trim() == open)?;
    let mut depth = 1;
    let mut body = Vec::new();
    for line in lines {
        let t = line.trim();
        if ["{{- if ", "{{- range ", "{{- with ", "{{- define "]
            .iter()
            .any(|p| t.starts_with(p))
        {
            depth += 1;
        } else if t.starts_with("{{- end") {
            depth -= 1;
            if depth == 0 {
                return Some(body);
            }
        }
        body.push(line);
    }
    None
}

/// The scalar at a dotted path of a YAML file, as written
/// (`a.b` is `b:` nested two spaces under `a:`).
fn yaml_scalar(text: &str, path: &str) -> Option<String> {
    let depth = |l: &str| l.len() - l.trim_start().len();
    let mut scope = text.to_string();
    let mut indent = 0;
    let mut keys = path.split('.').peekable();
    while let Some(key) = keys.next() {
        if keys.peek().is_none() {
            let head = format!("{key}:");
            return scope
                .lines()
                .find(|l| depth(l) == indent && l.trim_start().starts_with(&head))
                .map(|l| l.trim_start()[head.len()..].trim().to_string());
        }
        scope = yaml_block(&scope, indent, key)?.join("\n");
        indent += 2;
    }
    None
}

#[test]
fn the_secret_helpers_read_what_they_claim() {
    let template = "{{- if .Values.a }}\n- name: A\n  valueFrom:\n    secretKeyRef:\n      \
                    name: s\n{{- end }}\n- name: B\n  value: x\n- name: C\n  valueFrom:\n    \
                    configMapKeyRef:\n";
    let entries = secret_ref_entries(template);
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries["A"], "{{- if .Values.a }}");
    let helpers = "{{- define \"x\" -}}\na: b\n{{- if .c }}\nd\n{{- end }}\n{{- end }}\ne\n";
    assert_eq!(
        template_define(helpers, "x").unwrap(),
        vec!["a: b", "{{- if .c }}", "d", "{{- end }}"]
    );
    assert!(template_define(helpers, "y").is_none());
    let yaml = "a: \"\"\nb:\n  a: 1\n  c:\n    d: []\n";
    assert_eq!(yaml_scalar(yaml, "a").as_deref(), Some("\"\""));
    assert_eq!(yaml_scalar(yaml, "b.a").as_deref(), Some("1"));
    assert_eq!(yaml_scalar(yaml, "b.c.d").as_deref(), Some("[]"));
    assert!(yaml_scalar(yaml, "c").is_none());
    let secrets = server_secret_variables();
    assert!(
        secrets.contains(&"CRABCROP_SECRET".to_string()),
        "{secrets:?}"
    );
}

#[test]
fn the_helm_chart_reads_every_secret_variable_from_a_secret() {
    // Issue #589: the chart read only CRABCROP_SECRET and
    // CRABCROP_METRICS_TOKEN from a Secret. Every other secret
    // went through `env`, which writes it into the Deployment
    // as plain text, readable by anyone who can read the
    // Deployment or its Pods.
    let deployment = read("deploy/helm/crabcrop/templates/deployment.yaml");
    let entries = secret_ref_entries(&deployment);
    let secrets = server_secret_variables();
    let wanted = secrets
        .iter()
        .map(String::as_str)
        .chain(SECRET_PARTNERS.iter().map(|(id, _)| *id));
    let missing: Vec<&str> = wanted.filter(|n| !entries.contains_key(*n)).collect();
    assert!(
        missing.is_empty(),
        "deployment.yaml: no secretKeyRef entry for {missing:?}"
    );
    for (id, secret) in SECRET_PARTNERS {
        assert_eq!(
            entries[id], entries[secret],
            "deployment.yaml: {id} comes from the same Secret as {secret}"
        );
    }

    // The chart's route map names, for each secret, the value
    // that turns its entry on. The plain-value warning points
    // there, so it must match the server's list and the
    // template.
    let helpers = read("deploy/helm/crabcrop/templates/_helpers.tpl");
    let routes: std::collections::BTreeMap<&str, &str> =
        template_define(&helpers, "crabcrop.secretRoutes")
            .expect("_helpers.tpl: defines crabcrop.secretRoutes")
            .into_iter()
            .map(|l| l.split_once(": ").expect("NAME: value"))
            .collect();
    let mut listed: Vec<&str> = routes.keys().copied().collect();
    let mut server: Vec<&str> = secrets.iter().map(String::as_str).collect();
    listed.sort_unstable();
    server.sort_unstable();
    assert_eq!(
        listed, server,
        "crabcrop.secretRoutes lists SECRET_VARIABLES"
    );
    let values = read("deploy/helm/crabcrop/values.yaml");
    for (name, route) in routes {
        let guard = entries[name].replace(['(', ')'], "");
        assert!(
            guard.contains(&format!(".Values.{route}")),
            "deployment.yaml: {name} renders under {route}, not {guard:?}"
        );
        assert_eq!(
            yaml_scalar(&values, route).as_deref(),
            Some("\"\""),
            "values.yaml: {route} is a value, empty by default"
        );
    }
}

#[test]
fn the_helm_chart_takes_env_entries_and_whole_secrets() {
    // Issue #589: the `env` map renders every entry as a plain
    // string, so a `valueFrom` in it became the literal text
    // "map[valueFrom:...]". `extraEnv` takes entries as
    // written and `envFrom` maps whole Secrets.
    let values = read("deploy/helm/crabcrop/values.yaml");
    for key in ["extraEnv", "envFrom"] {
        assert_eq!(
            yaml_scalar(&values, key).as_deref(),
            Some("[]"),
            "values.yaml: {key} is a list, empty by default"
        );
    }
    let deployment = read("deploy/helm/crabcrop/templates/deployment.yaml");
    let lines: Vec<&str> = deployment.lines().map(str::trim).collect();
    let extra = lines
        .iter()
        .position(|l| *l == "{{- with .Values.extraEnv }}")
        .expect("deployment.yaml: renders extraEnv");
    assert_eq!(lines[extra + 1], "{{- toYaml . | nindent 12 }}");
    let listed = lines
        .iter()
        .position(|l| l.contains("range $k := keys $allEnv"))
        .unwrap();
    assert!(listed < extra, "extraEnv follows the env map");
    let from = lines
        .iter()
        .position(|l| *l == "{{- with .Values.envFrom }}")
        .expect("deployment.yaml: renders envFrom");
    assert_eq!(
        lines[from + 1..from + 3],
        ["envFrom:", "{{- toYaml . | nindent 12 }}"]
    );

    // extraEnv cannot carry the drain grace past its guard
    // (issue #463) either.
    let guard = lines
        .iter()
        .position(|l| l.contains(r#"eq (get . "name") "CRABCROP_SHUTDOWN_GRACE_SECS""#))
        .expect("deployment.yaml: refuses CRABCROP_SHUTDOWN_GRACE_SECS in extraEnv");
    assert!(
        lines[guard + 1].starts_with("{{- fail "),
        "{}",
        lines[guard + 1]
    );
    let range = lines[..guard]
        .iter()
        .rposition(|l| l.starts_with("{{- range "))
        .unwrap();
    assert_eq!(lines[range], "{{- range .Values.extraEnv }}");
}

#[test]
fn the_helm_chart_warns_when_a_secret_variable_is_a_plain_value() {
    // The owner's call on #589: for one release a secret in
    // `env`, `ai.env` or a plain `extraEnv` value still
    // renders, with a warning in the install notes and in the
    // manifest; `refusePlainSecrets` fails the render instead.
    let helpers = read("deploy/helm/crabcrop/templates/_helpers.tpl");
    let plain = template_define(&helpers, "crabcrop.plainSecrets")
        .expect("_helpers.tpl: defines crabcrop.plainSecrets")
        .join("\n");
    for read_from in [
        "include \"crabcrop.secretRoutes\" .",
        ".Values.env",
        ".Values.ai.env",
        ".Values.extraEnv",
    ] {
        assert!(
            plain.contains(read_from),
            "crabcrop.plainSecrets reads {read_from}"
        );
    }

    let include = "{{- $plainSecrets := include \"crabcrop.plainSecrets\" . | fromYaml }}";
    let notes = read("deploy/helm/crabcrop/templates/NOTES.txt");
    let notes: Vec<&str> = notes.lines().map(str::trim).collect();
    let at = notes
        .iter()
        .position(|l| *l == include)
        .expect("NOTES.txt: reads crabcrop.plainSecrets");
    assert_eq!(notes[at + 1], "{{- if $plainSecrets }}");
    assert!(
        notes[at + 2..].iter().any(|l| l.starts_with("WARNING: ")),
        "NOTES.txt: warns about plain secrets"
    );

    let deployment = read("deploy/helm/crabcrop/templates/deployment.yaml");
    let lines: Vec<&str> = deployment.lines().map(str::trim).collect();
    let at = lines
        .iter()
        .position(|l| *l == include)
        .expect("deployment.yaml: reads crabcrop.plainSecrets");
    assert_eq!(
        lines[at + 1..at + 3],
        [
            "{{- if and $plainSecrets .Values.refusePlainSecrets }}",
            "{{- $refused := list }}"
        ],
        "deployment.yaml: fails only behind refusePlainSecrets"
    );
    let comment = lines
        .iter()
        .position(|l| l.starts_with("# WARNING: {{ $name }} "))
        .expect("deployment.yaml: a comment names each plain secret");
    assert!(at < comment, "the comment follows the refusal");
    // The env map and extraEnv render after the Secret routes,
    // so a plain entry left there wins over the Secret's.
    assert!(
        lines[comment].contains("remove it from env"),
        "deployment.yaml: the comment says to remove the plain entry"
    );
    assert_eq!(
        lines[comment - 1],
        "{{- range $name, $route := $plainSecrets }}"
    );
    let values = read("deploy/helm/crabcrop/values.yaml");
    assert_eq!(
        yaml_scalar(&values, "refusePlainSecrets").as_deref(),
        Some("false"),
        "values.yaml: refusePlainSecrets is off for this release"
    );
}

#[test]
fn the_secret_docs_describe_routes_that_work() {
    // values.yaml and the chart README said the AWS keys could
    // go through the `env` map "backed by a Secret", which the
    // template could not do.
    let values = read("deploy/helm/crabcrop/values.yaml");
    assert!(
        !values.contains("through the env map"),
        "values.yaml: sends the AWS keys through env"
    );
    let readme = read("deploy/helm/crabcrop/README.md");
    assert!(
        !readme.contains("through the chart's `env` map"),
        "README.md: sends the AWS keys through env"
    );
    for route in [
        "resultBucket.credentialsSecret",
        "previousSecret.existingSecret",
        "gcsCredentials.existingSecret",
        "extraEnv",
        "envFrom",
        "refusePlainSecrets",
    ] {
        assert!(readme.contains(&format!("`{route}")), "README.md: {route}");
    }
    // The kustomize env patch shows the AWS keys as Secret
    // references, never as `value:` lines.
    let patch = read("deploy/kustomize/overlays/multi-replica/env-patch.yaml");
    let commented: Vec<&str> = patch
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("# "))
        .map(str::trim)
        .collect();
    for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] {
        let at = commented
            .iter()
            .position(|l| *l == format!("- name: {key}"))
            .unwrap_or_else(|| panic!("env-patch.yaml: a commented {key} entry"));
        assert_eq!(
            commented.get(at + 1..at + 3),
            Some(&["valueFrom:", "secretKeyRef:"][..]),
            "env-patch.yaml: {key} reads a Secret"
        );
    }
}

/// The three images, each with its own Dockerfile.
const DOCKERFILES: [&str; 3] = ["Dockerfile", "Dockerfile.release", "Dockerfile.ai"];

/// The images' health check: the binary probes its own
/// server, so the image needs no HTTP client.
const HEALTHCHECK: &str = r#"HEALTHCHECK --interval=30s --timeout=3s CMD ["crabcrop", "health"]"#;

#[test]
fn images_run_as_a_numeric_user() {
    // Issue #592: the images said `USER nobody`. The kubelet
    // cannot check a user name against `runAsNonRoot`, so a
    // pod that set only that never started ("image has
    // non-numeric user (nobody), cannot verify user is
    // non-root"). The AI image switches to root for its
    // payload steps and must end on the same user.
    for rel in DOCKERFILES {
        let stage = final_stage(&read(rel));
        let users: Vec<&str> = stage
            .iter()
            .map(String::as_str)
            .filter(|i| i.starts_with("USER "))
            .collect();
        assert_eq!(users.last(), Some(&RUNTIME_USER), "{rel}: {users:?}");
    }
}

#[test]
fn runtime_images_carry_no_curl() {
    // Issue #592: both runtime images installed curl only for
    // their HEALTHCHECK. The binary probes itself now
    // (`crabcrop health`), and the AI image downloads ONNX
    // Runtime in a stage of its own.
    for rel in DOCKERFILES {
        let stage = final_stage(&read(rel));
        let curl: Vec<&String> = stage.iter().filter(|i| i.contains("curl")).collect();
        assert!(
            curl.is_empty(),
            "{rel}: the runtime stage uses curl: {curl:?}"
        );
    }
    for rel in ["Dockerfile", "Dockerfile.release"] {
        let stage = final_stage(&read(rel));
        let checks: Vec<&str> = stage
            .iter()
            .map(String::as_str)
            .filter(|i| i.starts_with("HEALTHCHECK"))
            .collect();
        assert_eq!(checks, [HEALTHCHECK], "{rel}");
    }
    // The check runs a verb the CLI knows.
    let usage = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(crabcrop::cli::exec(&["--help".to_string()]))
        .unwrap();
    assert!(
        usage.contains("crabcrop health "),
        "the CLI lists no health verb:\n{usage}"
    );

    // The AI image keeps the base image's check. ONNX Runtime
    // comes from a download stage that checks both hashes;
    // the runtime stage copies only the library.
    let ai = stages(&read("Dockerfile.ai"));
    let [download, runtime] = &ai[..] else {
        panic!("Dockerfile.ai: a download stage, then the runtime stage: {ai:#?}");
    };
    assert!(download[0].ends_with(" AS onnxruntime"), "{}", download[0]);
    let fetch = download
        .iter()
        .find(|i| i.contains("curl -fsSL "))
        .expect("Dockerfile.ai: the download stage fetches ONNX Runtime");
    assert_eq!(
        fetch.matches("sha256sum -c").count(),
        2,
        "Dockerfile.ai: the tarball and the library are both checked: {fetch}"
    );
    assert!(
        runtime
            .iter()
            .any(|i| i == "COPY --from=onnxruntime /opt/onnxruntime /opt/onnxruntime"),
        "Dockerfile.ai: {runtime:#?}"
    );
    assert!(
        !runtime.iter().any(|i| i.starts_with("HEALTHCHECK")),
        "Dockerfile.ai: inherits the base image's check"
    );

    // Every Debian base, the download stage's included, is
    // the one pinned digest.
    let bases: std::collections::BTreeSet<String> = DOCKERFILES
        .iter()
        .flat_map(|rel| stages(&read(rel)))
        .filter_map(|stage| {
            stage[0]
                .split_whitespace()
                .find(|word| word.starts_with("debian:"))
                .map(String::from)
        })
        .collect();
    assert_eq!(bases.len(), 1, "one pinned Debian base: {bases:?}");
}

/// Every file under `dir`, recursively, sorted.
fn files_under(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(files_under(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// The lines of a text with continuation lines (a trailing
/// `\`) joined, each line's comment markers (`#`) and
/// indentation dropped, so a command split over commented
/// lines reads as one.
fn commands(text: &str) -> Vec<String> {
    let mut commands = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        let line = line.trim_start().trim_start_matches('#').trim();
        match line.strip_suffix('\\') {
            Some(head) => {
                current.push_str(head.trim_end());
                current.push(' ');
            }
            None => {
                current.push_str(line);
                commands.push(std::mem::take(&mut current));
            }
        }
    }
    commands
}

#[test]
fn no_recipe_runs_curl_inside_the_app_container() {
    // The images carry no curl (issue #592), so a check that
    // runs it in the app container (`docker compose exec
    // crabcrop curl ...`, `kubectl exec ... curl`) fails. Each
    // inline code span is read on its own.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = files_under(&root.join("deploy"));
    files.extend(
        files_under(&root.join("docs"))
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "md")),
    );
    files.extend(["README.md", "SECURITY.md", "CONTRIBUTING.md"].map(|f| root.join(f)));
    let mut found = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        for command in commands(&text) {
            for span in command.split('`') {
                let exec = span
                    .find("exec crabcrop")
                    .or_else(|| span.find("kubectl exec"));
                if exec.is_some_and(|at| span[at..].contains("curl")) {
                    let rel = path.strip_prefix(root).unwrap().display().to_string();
                    found.push(format!("{rel}: {span}"));
                }
            }
        }
    }
    assert!(found.is_empty(), "{found:#?}");

    // The reading is not vacuous.
    assert_eq!(
        commands("#   docker compose exec crabcrop \\\n#     curl -s x\nnext\n"),
        ["docker compose exec crabcrop curl -s x", "next"]
    );
}

/// The settings Pod Security "restricted" needs on the pod,
/// with the `fsGroup` the disk tier needs, as paths under
/// the pod `securityContext`.
const POD_SECURITY: [(&str, &str); 5] = [
    ("runAsNonRoot", "true"),
    ("runAsUser", "65534"),
    ("runAsGroup", "65534"),
    ("fsGroup", "65534"),
    ("seccompProfile.type", "RuntimeDefault"),
];

/// The container `securityContext`: what "restricted" needs,
/// and a read-only root filesystem.
const CONTAINER_SECURITY: [(&str, &str); 3] = [
    ("allowPrivilegeEscalation", "false"),
    ("readOnlyRootFilesystem", "true"),
    ("capabilities.drop", r#"["ALL"]"#),
];

/// `lines`, a block [`yaml_block`] returned, as a text whose
/// keys start at column 0.
fn dedent(lines: &[&str]) -> String {
    let depth = |l: &str| l.len() - l.trim_start().len();
    let by = lines.iter().map(|l| depth(l)).min().unwrap_or(0);
    lines
        .iter()
        .map(|l| &l[by..])
        .collect::<Vec<_>>()
        .join("\n")
}

/// How deep in `if`, `range` and `with` blocks line `at` of
/// a chart template sits: 0 renders whatever the values.
fn template_depth(lines: &[&str], at: usize) -> i64 {
    let count = |l: &str, words: &[&str]| -> i64 {
        words
            .iter()
            .map(|w| {
                l.matches(&format!("{{{{- {w}")).count() + l.matches(&format!("{{{{ {w}")).count()
            })
            .sum::<usize>() as i64
    };
    lines[..at]
        .iter()
        .map(|l| count(l, &["if ", "range ", "with "]) - count(l, &["end"]))
        .sum()
}

/// The index of the line `- name: <name>` whose next line,
/// trimmed, is `next`.
fn named_entry(lines: &[&str], name: &str, next: &str) -> Option<usize> {
    (0..lines.len().saturating_sub(1))
        .find(|&i| lines[i].trim() == format!("- name: {name}") && lines[i + 1].trim() == next)
}

#[test]
fn kubernetes_recipes_meet_pod_security_restricted() {
    // Issue #592: Pod Security "restricted" refused the pods
    // (no runAsNonRoot, no seccomp profile), and they skipped
    // the usual containment: the root filesystem was writable
    // and the namespace's service account token was mounted.
    // The server writes outside its configured directories
    // only in /tmp, so that is an emptyDir.
    let values = read("deploy/helm/crabcrop/values.yaml");
    for (path, want) in POD_SECURITY {
        assert_eq!(
            yaml_scalar(&values, &format!("podSecurityContext.{path}")).as_deref(),
            Some(want),
            "values.yaml: podSecurityContext.{path}"
        );
    }
    for (path, want) in CONTAINER_SECURITY {
        assert_eq!(
            yaml_scalar(&values, &format!("securityContext.{path}")).as_deref(),
            Some(want),
            "values.yaml: securityContext.{path}"
        );
    }
    assert_eq!(
        yaml_scalar(&values, "automountServiceAccountToken").as_deref(),
        Some("false"),
        "values.yaml: automountServiceAccountToken"
    );
    let size =
        yaml_scalar(&values, "tmpVolume.sizeLimit").expect("values.yaml: tmpVolume.sizeLimit");

    // The template renders the token setting and /tmp whatever
    // the values, a release upgraded with --reuse-values
    // included (its values lack them).
    let deployment = read("deploy/helm/crabcrop/templates/deployment.yaml");
    let lines: Vec<&str> = deployment.lines().collect();
    let token = lines
        .iter()
        .position(|l| {
            l.trim()
                == "automountServiceAccountToken: {{ .Values.automountServiceAccountToken | default false }}"
        })
        .expect("deployment.yaml: sets automountServiceAccountToken from the value");
    let mount = named_entry(&lines, "tmp", "mountPath: /tmp")
        .expect("deployment.yaml: mounts the tmp volume at /tmp");
    let volume = named_entry(&lines, "tmp", "emptyDir:")
        .expect("deployment.yaml: the tmp volume is an emptyDir");
    assert_eq!(
        lines[volume + 2].trim(),
        r#"sizeLimit: {{ (.Values.tmpVolume).sizeLimit | default "1Gi" }}"#,
        "deployment.yaml: the tmp volume's size"
    );
    for at in [token, mount, volume] {
        assert_eq!(
            template_depth(&lines, at),
            0,
            "deployment.yaml:{}: renders under a condition",
            at + 1
        );
    }
    assert_eq!(size, "1Gi", "values.yaml: the template's default size");

    // The kustomize base sets the same.
    let base = read("deploy/kustomize/base/deployment.yaml");
    let lines: Vec<&str> = base.lines().collect();
    let depth = |l: &str| l.len() - l.trim_start().len();
    let containers = lines
        .iter()
        .position(|l| l.trim() == "containers:")
        .expect("kustomize base: containers");
    let pod_indent = depth(lines[containers]);
    let pod =
        dedent(&yaml_block(&base, pod_indent, "securityContext").expect("pod securityContext"));
    for (path, want) in POD_SECURITY {
        assert_eq!(
            yaml_scalar(&pod, path).as_deref(),
            Some(want),
            "kustomize base: pod securityContext.{path}"
        );
    }
    let container_indent = depth(lines[containers + 1]) + 2;
    let container = dedent(
        &yaml_block(&base, container_indent, "securityContext").expect("container securityContext"),
    );
    for (path, want) in CONTAINER_SECURITY {
        assert_eq!(
            yaml_scalar(&container, path).as_deref(),
            Some(want),
            "kustomize base: container securityContext.{path}"
        );
    }
    assert!(
        lines
            .iter()
            .any(|l| depth(l) == pod_indent && l.trim() == "automountServiceAccountToken: false"),
        "kustomize base: automountServiceAccountToken: false on the pod"
    );
    named_entry(&lines, "tmp", "mountPath: /tmp").expect("kustomize base: mounts /tmp");
    let volume = named_entry(&lines, "tmp", "emptyDir:").expect("kustomize base: a tmp emptyDir");
    assert_eq!(
        lines[volume + 2].trim(),
        format!("sizeLimit: {size}"),
        "kustomize base: the chart's tmp size"
    );

    // Nothing layered on top takes the settings back.
    let keys = [
        "securityContext",
        "podSecurityContext",
        "runAsNonRoot",
        "runAsUser",
        "runAsGroup",
        "seccompProfile",
        "readOnlyRootFilesystem",
        "allowPrivilegeEscalation",
        "automountServiceAccountToken",
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut layered = yaml_files(&root.join("deploy/kustomize/components"));
    layered.extend(yaml_files(&root.join("deploy/kustomize/overlays")));
    layered.extend(
        ["values-production.yaml", "values-multireplica.yaml"]
            .map(|f| root.join("deploy/helm/crabcrop").join(f)),
    );
    for path in layered {
        let text = std::fs::read_to_string(&path).unwrap();
        let found = yaml_keys(&text, &keys);
        assert!(found.is_empty(), "{}: {found:?}", path.display());
    }
}

#[test]
fn kubernetes_recipes_request_modest_resources() {
    // Issue #592: the chart's default was `resources: {}`,
    // so a default install ran BestEffort, the first class
    // the kubelet evicts. The owner chose modest requests
    // and no limits by default; values-production.yaml sets
    // limits.
    let values = read("deploy/helm/crabcrop/values.yaml");
    let base = read("deploy/kustomize/base/deployment.yaml");
    let resources =
        dedent(&yaml_block(&base, 10, "resources").expect("kustomize base: container resources"));
    for (path, want) in [("requests.cpu", "100m"), ("requests.memory", "128Mi")] {
        assert_eq!(
            yaml_scalar(&values, &format!("resources.{path}")).as_deref(),
            Some(want),
            "values.yaml: resources.{path}"
        );
        assert_eq!(
            yaml_scalar(&resources, path).as_deref(),
            Some(want),
            "kustomize base: resources.{path}"
        );
    }
    assert!(yaml_scalar(&values, "resources.limits").is_none());
    assert!(yaml_scalar(&resources, "limits").is_none());
    let production = read("deploy/helm/crabcrop/values-production.yaml");
    for path in ["limits.cpu", "limits.memory"] {
        assert!(
            yaml_scalar(&production, &format!("resources.{path}")).is_some(),
            "values-production.yaml: resources.{path}"
        );
    }
}

/// The ranges the optional NetworkPolicy keeps the pod from
/// reaching: the private ranges, link-local (the cloud
/// metadata address) and carrier-grade NAT.
const PRIVATE_RANGES: [&str; 5] = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "100.64.0.0/10",
];

/// The items of the list under the first `key:` line, at any
/// indent: the `- ` lines at the depth of the first line
/// below the key, quotes dropped. Comments are skipped.
fn list_under(text: &str, key: &str) -> Vec<String> {
    let depth = |l: &str| l.len() - l.trim_start().len();
    let mut lines = text.lines();
    let Some(head) = lines.find(|l| l.trim() == format!("{key}:")) else {
        return Vec::new();
    };
    let block: Vec<&str> = lines
        .take_while(|l| l.trim().is_empty() || depth(l) > depth(head))
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect();
    let Some(items) = block.first().map(|l| depth(l)) else {
        return Vec::new();
    };
    block
        .iter()
        .filter(|l| depth(l) == items)
        .filter_map(|l| l.trim().strip_prefix("- "))
        .map(|item| item.trim_matches('"').to_string())
        .collect()
}

#[test]
fn the_pod_helpers_read_what_they_claim() {
    let yaml = "a:\n  # c\n  - x\n  - y:\n      z:\n        - q\n  - \"w\"\nb:\n  - v\n";
    assert_eq!(list_under(yaml, "a"), ["x", "y:", "w"]);
    assert_eq!(list_under(yaml, "z"), ["q"]);
    assert_eq!(list_under(yaml, "b"), ["v"]);
    assert!(list_under(yaml, "c").is_empty());
    assert_eq!(dedent(&["    a:", "      b: 1"]), "a:\n  b: 1");
    let template = [
        "{{- if .a }}",
        "x",
        "{{- end }}",
        "y: \"{{ if .b }}1{{ else }}2{{ end }}\"",
        "z",
    ];
    assert_eq!(template_depth(&template, 1), 1);
    assert_eq!(template_depth(&template, 4), 0);
    let entries = [
        "- name: tmp",
        "  emptyDir:",
        "- name: tmp",
        "  mountPath: /tmp",
    ];
    assert_eq!(named_entry(&entries, "tmp", "mountPath: /tmp"), Some(2));
    assert_eq!(named_entry(&entries, "tmp", "emptyDir:"), Some(0));
    assert_eq!(named_entry(&entries, "x", "emptyDir:"), None);
}

#[test]
fn the_network_policy_is_optional_and_allows_public_egress() {
    // Issue #592: no NetworkPolicy shipped. The in-app check
    // (CRABCROP_BLOCK_PRIVATE_SOURCES) skips hosts in the
    // allowlist, and code running in the pod is not bound by
    // it at all. The policy is off by default in the chart
    // and a component in kustomize.
    let values = read("deploy/helm/crabcrop/values.yaml");
    assert_eq!(
        yaml_scalar(&values, "networkPolicy.enabled").as_deref(),
        Some("false")
    );
    for key in ["ingressFrom", "extraEgress"] {
        assert_eq!(
            yaml_scalar(&values, &format!("networkPolicy.{key}")).as_deref(),
            Some("[]"),
            "values.yaml: networkPolicy.{key}"
        );
    }
    let helm = read("deploy/helm/crabcrop/templates/networkpolicy.yaml");
    let lines: Vec<&str> = helm.lines().collect();
    assert_eq!(
        lines.first(),
        Some(&"{{- if (.Values.networkPolicy).enabled }}"),
        "networkpolicy.yaml: renders only when enabled, from values kept by --reuse-values too"
    );
    assert_eq!(lines.last(), Some(&"{{- end }}"));
    let kustomize = read("deploy/kustomize/components/network-policy/networkpolicy.yaml");
    for (rel, policy) in [("helm", &helm), ("kustomize", &kustomize)] {
        assert!(policy.contains("kind: NetworkPolicy"), "{rel}");
        assert_eq!(
            list_under(policy, "policyTypes"),
            ["Ingress", "Egress"],
            "{rel}"
        );
        // Ingress: the HTTP port only.
        assert_eq!(
            list_under(policy, "ingress").len(),
            1,
            "{rel}: one ingress rule"
        );
        assert!(
            policy.contains("- port: http\n"),
            "{rel}: ingress to the http port"
        );
        // Egress: DNS, then public addresses.
        assert_eq!(policy.matches("- port: 53\n").count(), 2, "{rel}: DNS");
        for protocol in ["UDP", "TCP"] {
            assert!(policy.contains(&format!("protocol: {protocol}")), "{rel}");
        }
        assert!(policy.contains("cidr: 0.0.0.0/0\n"), "{rel}");
        assert_eq!(list_under(policy, "except"), PRIVATE_RANGES, "{rel}");
    }
    let component = read("deploy/kustomize/components/network-policy/kustomization.yaml");
    assert!(component.contains("kind: Component"));
    assert_eq!(list_under(&component, "resources"), ["networkpolicy.yaml"]);
    let full = read("deploy/kustomize/overlays/full/kustomization.yaml");
    assert!(full.contains("- ../../components/network-policy\n"));
    let base = read("deploy/kustomize/base/kustomization.yaml");
    assert!(!base.contains("network"), "the base leaves the policy out");
}
