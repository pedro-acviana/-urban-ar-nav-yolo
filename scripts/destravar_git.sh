#!/usr/bin/env bash
# Remove locks pendentes do Git com segurança.
#
# Só é necessário quando o repositório é acessado de um ambiente que não
# consegue apagar arquivos (montagens de rede, contêineres). Nesses casos o
# Git deixa para trás .lock que ele mesmo não conseguiu limpar, e qualquer
# comando seguinte falha com "Another git process seems to be running".
#
# ATENÇÃO: um .lock de referência **não vazio** contém o SHA que o Git estava
# prestes a gravar. Apagá-lo às cegas descarta a atualização e deixa o branch
# para trás sem aviso. Este script aplica o conteúdo em vez de perdê-lo.

set -euo pipefail

repo="${1:-.}"
cd "$repo"
git_dir="$(git rev-parse --git-dir)"

encontrados=0

while IFS= read -r -d '' lock; do
    encontrados=1
    alvo="${lock%.lock}"
    conteudo="$(tr -d '[:space:]' < "$lock" || true)"

    if [[ -n "$conteudo" && "$alvo" == *"/refs/"* ]]; then
        echo "aplicando atualização pendente: ${alvo#"$git_dir"/} -> $conteudo"
        cp "$lock" "$alvo"
    else
        echo "descartando lock vazio: ${lock#"$git_dir"/}"
    fi

    rm -f "$lock" 2>/dev/null || mv "$lock" "$git_dir/lixo_locks" 2>/dev/null || true
done < <(find "$git_dir" -name '*.lock' -print0)

[[ $encontrados -eq 0 ]] && echo "nenhum lock pendente"
exit 0
