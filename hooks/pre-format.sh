#!/usr/bin/env bash

# 8.3 Customizing Git - Git Hooks ; https://git-scm.com/book/gr/v2/Customizing-Git-Git-Hooks

# Add *pre-format.sh* to git's hook.
# ```bash
# ln -n hooks/pre-format.sh .git/hooks/pre-commit
# ```

# rustfmt § Quick start ; https://github.com/rust-lang-nursery/rustfmt#quick-start
# setup rustfmt(1) if uninstalled.
if ! command -v rustfmt >/dev/null; then
    rustup component add rustfmt-preview
fi

# All changed files will be reformated by rustfmt.
for file in $(git diff --name-only --cached); do
    if [ ${file: -3} == ".rs" ]; then
        # quiet option to suppress output.
        cargo fmt --quiet $file
    fi
done

exit 0
