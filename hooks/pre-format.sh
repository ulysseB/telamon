#! /bin/bash

# 8.3 Customizing Git - Git Hooks ; https://git-scm.com/book/gr/v2/Customizing-Git-Git-Hooks

# Add *pre-format.sh* to git's hook.
# ```bash
# ln -n hooks/pre-format.sh .git/hooks/pre-commit
# ```

# rustfmt § Installation https://github.com/rust-lang-nursery/rustfmt#installation
if ! command -v rustfmt >/dev/null; then
    cargo install rustfmt
fi

# quiet option to suppress output/
# don't reformat child modules.
for file in $(git diff --name-only --cached); do
    if [ ${file: -3} == ".rs" ]; then
        rustfmt --quiet --skip-children $file
    fi
done

exit 0
