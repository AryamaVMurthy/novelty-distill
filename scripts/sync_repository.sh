#!/bin/bash
# Synchronize a clean shared checkout, tolerating only bounded transient fetch failures.

set -euo pipefail

repo_dir="${1:?usage: sync_repository.sh REPO_DIR REPO_REF}"
repo_ref="${2:?usage: sync_repository.sh REPO_DIR REPO_REF}"
attempts="${REPOSITORY_FETCH_ATTEMPTS:-6}"
retry_seconds="${REPOSITORY_FETCH_RETRY_SECONDS:-10}"
for value in "${attempts}" "${retry_seconds}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "repository fetch attempts and retry seconds must be positive integers" >&2
    exit 2
  fi
done
if [[ ! -d "${repo_dir}/.git" ]]; then
  echo "repository checkout is missing: ${repo_dir}" >&2
  exit 1
fi
if [[ -n "$(git -C "${repo_dir}" status --porcelain)" ]]; then
  echo "repository checkout has local changes: ${repo_dir}" >&2
  exit 1
fi

fetched=0
for ((attempt = 1; attempt <= attempts; attempt++)); do
  if git -C "${repo_dir}" fetch origin "${repo_ref}"; then
    fetched=1
    break
  fi
  echo "repository fetch attempt ${attempt}/${attempts} failed for ${repo_ref}" >&2
  if ((attempt < attempts)); then
    sleep "${retry_seconds}"
  fi
done
if [[ "${fetched}" != 1 ]]; then
  echo "repository fetch exhausted ${attempts} attempts; refusing stale-code fallback" >&2
  exit 1
fi
git -C "${repo_dir}" switch "${repo_ref}"
git -C "${repo_dir}" merge --ff-only "origin/${repo_ref}"
echo "repository_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
