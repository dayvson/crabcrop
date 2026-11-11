{{/*
Expand the name of the chart.
*/}}
{{- define "crabcrop.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name.
*/}}
{{- define "crabcrop.fullname" -}}
{{- if contains .Chart.Name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "crabcrop.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "crabcrop.labels" -}}
helm.sh/chart: {{ include "crabcrop.chart" . }}
{{ include "crabcrop.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "crabcrop.selectorLabels" -}}
app.kubernetes.io/name: {{ include "crabcrop.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* The Secret holding CRABCROP_SECRET. */}}
{{- define "crabcrop.secretName" -}}
{{- if .Values.existingSecret }}
{{- .Values.existingSecret }}
{{- else }}
{{- include "crabcrop.fullname" . }}
{{- end }}
{{- end }}

{{/*
The variables the server treats as secrets (SECRET_VARIABLES
in src/config.rs), each with the value that reads it from a
Secret. tests/deploy_recipes.rs keeps this list equal to the
server's.
*/}}
{{- define "crabcrop.secretRoutes" -}}
CRABCROP_SECRET: existingSecret
CRABCROP_SECRET_PREVIOUS: previousSecret.existingSecret
AWS_SECRET_ACCESS_KEY: resultBucket.credentialsSecret
AWS_SESSION_TOKEN: resultBucket.credentialsSecret
CRABCROP_GCS_SECRET_KEY: gcsCredentials.existingSecret
CRABCROP_METRICS_TOKEN: metricsToken.existingSecret
{{- end }}

{{/*
The secrets the Deployment would carry as plain values: those
in env, in ai.env when ai.enabled, and extraEnv entries with
value:. As YAML, each name mapped to the value to set instead;
empty when there are none.
*/}}
{{- define "crabcrop.plainSecrets" -}}
{{- $routes := include "crabcrop.secretRoutes" . | fromYaml }}
{{- $names := keys (.Values.env | default dict) }}
{{- if .Values.ai.enabled }}
{{- $names = concat $names (keys (.Values.ai.env | default dict)) }}
{{- end }}
{{- range .Values.extraEnv }}
{{- if hasKey . "value" }}
{{- $names = append $names (get . "name") }}
{{- end }}
{{- end }}
{{- range $name := $names | uniq | sortAlpha }}
{{- if hasKey $routes $name }}
{{ $name }}: {{ get $routes $name }}
{{- end }}
{{- end }}
{{- end }}
