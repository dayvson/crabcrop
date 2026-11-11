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

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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
fn yaml_files(dir: &Path) -> Vec<PathBuf> {
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

/// The variables that hold a secret the server checks at
/// boot (issue #588).
const SECRET_VARS: [&str; 3] = [
    "CRABCROP_SECRET",
    "CRABCROP_SECRET_PREVIOUS",
    "CRABCROP_METRICS_TOKEN",
];

/// The value at the start of `rest`: a quoted string's
/// contents, or the text up to white space, a quote or a
/// backtick.
fn leading_value(rest: &str) -> String {
    for quote in ['"', '\''] {
        if let Some(inner) = rest.strip_prefix(quote) {
            return inner.split(quote).next().unwrap_or("").to_string();
        }
    }
    rest.split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '`'))
        .next()
        .unwrap_or("")
        .to_string()
}

/// The trimmed lines of each YAML list item `- name: <name>`
/// in `lines`, after its name line.
fn yaml_items<'a>(lines: &[&'a str], name: &str) -> Vec<Vec<&'a str>> {
    let mut items = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let Some(named) = t.strip_prefix("- name:") else {
            continue;
        };
        if named.trim().trim_matches(|c| c == '"' || c == '\'') != name {
            continue;
        }
        let indent = line.len() - t.len();
        items.push(
            lines[i + 1..]
                .iter()
                .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
                .take_while(|l| l.len() - l.trim_start().len() > indent)
                .map(|l| l.trim())
                .collect(),
        );
    }
    items
}

/// Every value `text` gives one of [`SECRET_VARS`], as
/// (variable, value): `NAME=value` (shell, env files,
/// `--from-literal`), `NAME: value` (YAML maps),
/// `"NAME": "value"` (JSON), and a YAML env item
/// `- name: NAME` with a `value:`, or with a `secretRef:`
/// to an item that has one (Container Apps).
fn secret_literals(text: &str) -> Vec<(&'static str, String)> {
    let is_name = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut found = Vec::new();
    for line in text.lines() {
        for var in SECRET_VARS {
            for (at, _) in line.match_indices(var) {
                if line[..at].chars().next_back().is_some_and(is_name) {
                    continue;
                }
                let rest = &line[at + var.len()..];
                if rest.chars().next().is_some_and(is_name) {
                    continue;
                }
                let value = if let Some(r) = rest.strip_prefix('=') {
                    Some(leading_value(r))
                } else if let Some(r) = rest.strip_prefix(": ") {
                    Some(leading_value(r.trim_start()))
                } else {
                    rest.strip_prefix('"')
                        .and_then(|r| r.trim_start().strip_prefix(':'))
                        .map(|r| leading_value(r.trim_start()))
                };
                if let Some(value) = value {
                    found.push((var, value));
                }
            }
        }
    }
    let lines: Vec<&str> = text.lines().collect();
    for var in SECRET_VARS {
        for item in yaml_items(&lines, var) {
            for key in item {
                if let Some(v) = key.strip_prefix("value:") {
                    found.push((var, leading_value(v.trim())));
                } else if let Some(name) = key.strip_prefix("secretRef:") {
                    for secret in yaml_items(&lines, name.trim()) {
                        for key in secret {
                            if let Some(v) = key.strip_prefix("value:") {
                                found.push((var, leading_value(v.trim())));
                            }
                        }
                    }
                }
            }
        }
    }
    found
}

#[test]
fn secret_literals_reads_every_spelling_the_docs_use() {
    let text = "\
CRABCROP_SECRET=a1 crabcrop
docker run -e CRABCROP_SECRET=\"a2\" -p 3000:3000
{\"CRABCROP_METRICS_TOKEN\": \"a3\"}
  CRABCROP_SECRET_PREVIOUS: a4
`CRABCROP_SECRET=a5` in prose
env:
  - name: CRABCROP_METRICS_TOKEN
    # a comment
    value: a6
  - name: CRABCROP_SECRET
    secretRef: signing
secrets:
  - name: signing
    value: a7
CRABCROP_SECRET_KEY_ID=no NUXT_CRABCROP_SECRET=no
CRABCROP_SECRET: \"${CRABCROP_SECRET:?CRABCROP_SECRET must be set}\"
Set `CRABCROP_SECRET` (16+ bytes) and CRABCROP_SECRET.
{\"name\":\"CRABCROP_SECRET\",\"valueFrom\":\"arn\"}
";
    let found = secret_literals(text);
    let found: Vec<(&str, &str)> = found
        .iter()
        .map(|(var, value)| (*var, value.as_str()))
        .collect();
    assert_eq!(
        found,
        [
            ("CRABCROP_SECRET", "a1"),
            ("CRABCROP_SECRET", "a2"),
            ("CRABCROP_METRICS_TOKEN", "a3"),
            ("CRABCROP_SECRET_PREVIOUS", "a4"),
            ("CRABCROP_SECRET", "a5"),
            (
                "CRABCROP_SECRET",
                "${CRABCROP_SECRET:?CRABCROP_SECRET must be set}"
            ),
            ("CRABCROP_SECRET", "a7"),
            ("CRABCROP_METRICS_TOKEN", "a6"),
        ]
    );
}

/// The text files the scan reads: the root README and
/// CONTRIBUTING, everything under `docs/` and `deploy/`, and
/// the READMEs of the SDKs, framework integrations, plugins
/// and crates, which an operator copies commands from too.
fn documented_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, readmes_only: bool, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if path.is_dir() {
                if name != "node_modules" && name != "target" && !name.starts_with('.') {
                    walk(&path, readmes_only, out);
                }
            } else if !readmes_only || name == "README.md" {
                out.push(path);
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = vec![root.join("README.md"), root.join("CONTRIBUTING.md")];
    for dir in ["docs", "deploy"] {
        walk(&root.join(dir), false, &mut files);
    }
    for dir in ["sdks", "frameworks", "plugins", "crates"] {
        walk(&root.join(dir), true, &mut files);
    }
    files.sort();
    files
}

/// Boot the binary with `var` set to `value` (and a real
/// signing secret, when `var` is another variable). `Ok`
/// when the boot is refused with an error that names `var`;
/// otherwise what happened instead.
fn boot_refuses(var: &str, value: &str) -> Result<(), String> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command.env_clear().env("PORT", "0");
    if var != "CRABCROP_SECRET" {
        command.env("CRABCROP_SECRET", "deploy-recipes-scan-4d1f0a9c2e7b");
    }
    let mut child = command
        .env(var, value)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (sender, lines) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = sender.send(line);
        }
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    let outcome = loop {
        if let Ok(line) = lines.recv_timeout(Duration::from_millis(50))
            && line.starts_with("crabcrop listening on port ")
        {
            break Err("it booted and listened".to_string());
        }
        if let Some(status) = child.try_wait().unwrap() {
            let mut stderr = String::new();
            let _ = child.stderr.take().unwrap().read_to_string(&mut stderr);
            break if status.code() == Some(1) && stderr.contains(var) {
                Ok(())
            } else {
                Err(format!("it exited with {status}: {stderr}"))
            };
        }
        if Instant::now() >= deadline {
            break Err("it neither exited nor listened within 30 s".to_string());
        }
    };
    let _ = child.kill();
    let _ = child.wait();
    outcome
}

#[test]
fn no_secret_value_in_the_docs_or_recipes_boots_the_server() {
    // Issue #588: the README, the VM and Container Apps
    // recipes and the compose metrics line shipped values the
    // server accepted, so a deployment that kept one ran with
    // a key anyone can read here. Every literal value given
    // to a secret variable must stop the boot now. Shell
    // substitutions (`$(openssl rand -hex 32)`, `${VAR}`) are
    // not literals, and neither is a `<placeholder>` in
    // Markdown, where it reads as a shell redirect.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut literals: Vec<(&str, String, Vec<String>)> = Vec::new();
    for path in documented_files() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let markdown = path.extension().is_some_and(|e| e == "md");
        let rel = path.strip_prefix(root).unwrap().display().to_string();
        for (var, value) in secret_literals(&text) {
            let placeholder = markdown && value.starts_with('<') && value.ends_with('>');
            if value.is_empty() || value.starts_with('$') || placeholder {
                continue;
            }
            match literals
                .iter_mut()
                .find(|(v, x, _)| *v == var && *x == value)
            {
                Some((_, _, files)) => files.push(rel.clone()),
                None => literals.push((var, value, vec![rel.clone()])),
            }
        }
    }
    // The scan reads what it claims: the docs do give the
    // CLI's signing secret as a literal.
    assert!(
        literals
            .iter()
            .any(|(_, _, files)| files.iter().any(|f| f == "docs/endpoints.md")),
        "{literals:?}"
    );
    let mut booted = Vec::new();
    for (var, value, files) in &literals {
        if let Err(what) = boot_refuses(var, value) {
            booted.push(format!("{var}={value:?} in {files:?}: {what}"));
        }
    }
    assert!(booted.is_empty(), "{booted:#?}");
}
