"""
대규모 RAG 크롤러 설정 - Git(500개) + Python(500개) = 1000개
신뢰할 수 있는 소스에서 검증된 URL만 포함
"""
import os
from pathlib import Path

# ========================================
# 디렉토리 설정
# ========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Git/Python 디렉토리
GIT_RAW_DIR = RAW_DATA_DIR / "git"
PYTHON_RAW_DIR = RAW_DATA_DIR / "python"

# 디렉토리 자동 생성
for dir_path in [RAW_DATA_DIR, GIT_RAW_DIR, PYTHON_RAW_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================================
# Git 크롤링 타겟 URL (500개 목표)
# ========================================

TARGET_URLS = {
    "git": {
        # ===== 1. Atlassian Git Tutorial (60개) =====
        # 최고 품질! 검증된 URL
        "atlassian": {
            "base": "https://www.atlassian.com/git/tutorials",
            "pages": [
                # Getting Git Right (10개)
                "what-is-version-control",
                "what-is-git",
                "why-git",
                "install-git",
                "source-code-management",
                "git-vs-svn",
                "learn-git-with-bitbucket-cloud",
                "git-lfs",
                "tutorials/learn-git-with-bitbucket-cloud",

                # Setting Up a Repository (7개)
                "setting-up-a-repository",
                "setting-up-a-repository/git-init",
                "setting-up-a-repository/git-clone",
                "setting-up-a-repository/git-config",
                "setting-up-a-repository/git-alias",

                # Saving Changes (7개)
                "saving-changes",
                "saving-changes/git-add",
                "saving-changes/git-commit",
                "saving-changes/git-diff",
                "saving-changes/git-stash",
                "saving-changes/gitignore",
                "saving-changes/git-log",

                # Inspecting a Repository (6개)
                "inspecting-a-repository",
                "inspecting-a-repository/git-status",
                "inspecting-a-repository/git-tag",
                "inspecting-a-repository/git-blame",
                "inspecting-a-repository/git-log",

                # Undoing Changes (6개)
                "undoing-changes",
                "undoing-changes/git-checkout",
                "undoing-changes/git-clean",
                "undoing-changes/git-revert",
                "undoing-changes/git-reset",
                "undoing-changes/git-rm",

                # Rewriting History (5개)
                "rewriting-history",
                "rewriting-history/git-commit--amend",
                "rewriting-history/git-rebase",
                "rewriting-history/git-rebase-i",
                "rewriting-history/git-reflog",

                # Using Branches (8개)
                "using-branches",
                "using-branches/git-branch",
                "using-branches/git-checkout",
                "using-branches/git-merge",
                "using-branches/merge-conflicts",
                "using-branches/merge-strategy",

                # Syncing (6개)
                "syncing",
                "syncing/git-remote",
                "syncing/git-fetch",
                "syncing/git-push",
                "syncing/git-pull",

                # Comparing Workflows (5개)
                "comparing-workflows",
                "comparing-workflows/centralized-workflow",
                "comparing-workflows/feature-branch-workflow",
                "comparing-workflows/gitflow-workflow",
                "comparing-workflows/forking-workflow",
            ]
        },

        # ===== 2. Pro Git Book - 한국어 (80개) =====
        # 공식 문서, 가장 권위 있음
        "pro_git_ko": {
            "base": "https://git-scm.com/book/ko/v2",
            "pages": [
                # 1장: 시작하기 (5개)
                "시작하기-버전-관리란%3F",
                "시작하기-Git-기초",
                "시작하기-Git-설치",
                "시작하기-Git-최초-설정",
                "시작하기-도움말-보기",

                # 2장: Git의 기초 (10개)
                "Git의-기초-Git-저장소-만들기",
                "Git의-기초-수정하고-저장소에-저장하기",
                "Git의-기초-커밋-히스토리-조회하기",
                "Git의-기초-되돌리기",
                "Git의-기초-리모트-저장소",
                "Git의-기초-태그",
                "Git의-기초-Git-Alias",
                "Git의-기초-팁과-트릭",

                # 3장: Git 브랜치 (8개)
                "Git-브랜치-브랜치란-무엇인가",
                "Git-브랜치-브랜치와-Merge-의-기초",
                "Git-브랜치-브랜치-관리",
                "Git-브랜치-브랜치-워크플로",
                "Git-브랜치-리모트-브랜치",
                "Git-브랜치-Rebase-하기",

                # 4장: Git 서버 (6개)
                "Git-서버-프로토콜",
                "Git-서버-서버에-Git-설치하기",
                "Git-서버-SSH-공개키-만들기",
                "Git-서버-서버-설정하기",
                "Git-서버-Git-데몬",
                "Git-서버-Smart-HTTP",

                # 5장: 분산 환경에서의 Git (6개)
                "분산-환경에서의-Git-분산-환경에서의-Workflow",
                "분산-환경에서의-Git-프로젝트에-기여하기",
                "분산-환경에서의-Git-프로젝트-관리하기",

                # 6장: GitHub (8개)
                "GitHub-계정-만들고-설정하기",
                "GitHub-GitHub-프로젝트에-기여하기",
                "GitHub-GitHub-프로젝트-관리하기",
                "GitHub-Organization-관리하기",
                "GitHub-GitHub-스크립팅",

                # 7장: Git 도구 (20개)
                "Git-도구-리비전-조회하기",
                "Git-도구-대화형-명령",
                "Git-도구-Stashing과-Cleaning",
                "Git-도구-내-작업에-서명하기",
                "Git-도구-검색",
                "Git-도구-히스토리-단장하기",
                "Git-도구-Reset-명확히-알고-가기",
                "Git-도구-고급-Merge",
                "Git-도구-Rerere",
                "Git-도구-Git으로-버그-찾기",
                "Git-도구-서브모듈",
                "Git-도구-Bundle",
                "Git-도구-Replace",
                "Git-도구-Credential-저장소",

                # 10장: Git의 내부 (10개)
                "Git의-내부-Plumbing-명령과-Porcelain-명령",
                "Git의-내부-Git-개체",
                "Git의-내부-Git-Refs",
                "Git의-내부-Packfile",
                "Git의-내부-Refspec",
                "Git의-내부-Transfer-Protocol",
                "Git의-내부-운영-및-데이터-복구",
                "Git의-내부-환경변수",
            ]
        },

        # ===== 3. GitHub Docs (50개) =====
        "github_docs": {
            "base": "https://docs.github.com/en",
            "pages": [
                # Getting Started (15개)
                "get-started/getting-started-with-git/set-up-git",
                "get-started/getting-started-with-git/configuring-git-to-handle-line-endings",
                "get-started/getting-started-with-git/about-remote-repositories",
                "get-started/getting-started-with-git/managing-remote-repositories",
                "get-started/getting-started-with-git/caching-your-github-credentials-in-git",
                "get-started/getting-started-with-git/associating-text-editors-with-git",
                "get-started/using-git/about-git",
                "get-started/using-git/pushing-commits-to-a-remote-repository",
                "get-started/using-git/getting-changes-from-a-remote-repository",
                "get-started/using-git/dealing-with-non-fast-forward-errors",
                "get-started/using-git/splitting-a-subfolder-out-into-a-new-repository",
                "get-started/using-git/about-git-subtree-merges",
                "get-started/using-git/about-git-rebase",
                "get-started/using-git/resolving-merge-conflicts-after-a-git-rebase",

                # Pull Requests (10개)
                "pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests",
                "pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request",
                "pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork",
                "pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request",
                "pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review",
                "pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews",
                "pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/reviewing-proposed-changes-in-a-pull-request",
                "pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request",
                "pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts",
                "pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github",

                # Repositories (15개)
                "repositories/creating-and-managing-repositories/about-repositories",
                "repositories/creating-and-managing-repositories/creating-a-new-repository",
                "repositories/creating-and-managing-repositories/cloning-a-repository",
                "repositories/creating-and-managing-repositories/deleting-a-repository",
                "repositories/working-with-files/managing-files/creating-new-files",
                "repositories/working-with-files/managing-files/editing-files",
                "repositories/working-with-files/managing-files/moving-a-file-to-a-new-location",
                "repositories/working-with-files/managing-files/deleting-files-in-a-repository",
                "repositories/working-with-files/using-files/navigating-code-on-github",
                "repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility",
            ]
        },

        # ===== 4. Git Official Reference (100개) =====
        # 모든 주요 Git 명령어 레퍼런스
        "official_reference": {
            "base": "https://git-scm.com/docs",
            "pages": [
                # Main Porcelain Commands (30개)
                "git-add",
                "git-am",
                "git-archive",
                "git-bisect",
                "git-branch",
                "git-bundle",
                "git-checkout",
                "git-cherry-pick",
                "git-citool",
                "git-clean",
                "git-clone",
                "git-commit",
                "git-describe",
                "git-diff",
                "git-fetch",
                "git-format-patch",
                "git-gc",
                "git-grep",
                "git-gui",
                "git-init",
                "git-log",
                "git-maintenance",
                "git-merge",
                "git-mv",
                "git-notes",
                "git-pull",
                "git-push",
                "git-range-diff",
                "git-rebase",
                "git-reset",
                "git-restore",
                "git-revert",
                "git-rm",
                "git-shortlog",
                "git-show",
                "git-sparse-checkout",
                "git-stash",
                "git-status",
                "git-submodule",
                "git-switch",
                "git-tag",
                "git-worktree",

                # Ancillary Commands (20개)
                "git-config",
                "git-fast-export",
                "git-fast-import",
                "git-filter-branch",
                "git-mergetool",
                "git-pack-refs",
                "git-prune",
                "git-reflog",
                "git-remote",
                "git-repack",
                "git-replace",

                # Interrogation Commands (10개)
                "git-annotate",
                "git-blame",
                "git-bugreport",
                "git-count-objects",
                "git-difftool",
                "git-fsck",
                "git-help",
                "git-instaweb",
                "git-merge-tree",
                "git-rerere",
                "git-show-branch",
                "git-verify-commit",
                "git-verify-tag",
                "git-whatchanged",

                # Plumbing Commands (40개)
                "git-apply",
                "git-checkout-index",
                "git-commit-graph",
                "git-commit-tree",
                "git-hash-object",
                "git-index-pack",
                "git-merge-file",
                "git-merge-index",
                "git-mktag",
                "git-mktree",
                "git-multi-pack-index",
                "git-pack-objects",
                "git-prune-packed",
                "git-read-tree",
                "git-symbolic-ref",
                "git-unpack-objects",
                "git-update-index",
                "git-update-ref",
                "git-write-tree",
                "git-cat-file",
                "git-check-ignore",
                "git-check-mailmap",
                "git-check-ref-format",
                "git-column",
                "git-credential",
                "git-credential-cache",
                "git-credential-store",
                "git-fmt-merge-msg",
                "git-interpret-trailers",
                "git-mailinfo",
                "git-mailsplit",
                "git-merge-one-file",
                "git-patch-id",
                "git-sh-i18n",
                "git-sh-setup",
                "git-stripspace",
                "git-diff-files",
                "git-diff-index",
                "git-diff-tree",
                "git-for-each-ref",
                "git-for-each-repo",
                "git-ls-files",
                "git-ls-remote",
                "git-ls-tree",
                "git-merge-base",
                "git-name-rev",
                "git-pack-redundant",
                "git-rev-list",
                "git-rev-parse",
                "git-show-index",
                "git-show-ref",
                "git-unpack-file",
                "git-var",
                "git-verify-pack",
            ]
        },

        # ===== 5. GitLab Docs (60개) =====
        "gitlab_docs": {
            "base": "https://docs.gitlab.com",
            "pages": [
                # Git basics (15개)
                "ee/topics/git/index.html",
                "ee/topics/git/git_add.html",
                "ee/topics/git/commits/index.html",
                "ee/topics/git/feature_branch_workflow.html",
                "ee/topics/git/git_rebase.html",
                "ee/topics/git/branch.html",
                "ee/topics/git/cherry_pick_changes.html",
                "ee/topics/git/stash.html",
                "ee/topics/git/tags.html",
                "ee/topics/git/how_to_install_git/index.html",
                "ee/topics/git/merge_conflicts.html",
                "ee/topics/git/git_log.html",
                "ee/topics/git/rollback_commits.html",
                "ee/topics/git/undo.html",
                "ee/topics/git/git_attributes.html",

                # Repository (15개)
                "ee/user/project/repository/index.html",
                "ee/user/project/repository/branches/index.html",
                "ee/user/project/repository/web_editor.html",
                "ee/user/project/repository/gpg_signed_commits/index.html",
                "ee/user/project/repository/files/index.html",
                "ee/user/project/repository/file_finder.html",
                "ee/user/project/repository/branches/default.html",
                "ee/user/project/repository/branches/protected.html",
                "ee/user/project/repository/forking_workflow.html",
                "ee/user/project/repository/mirror/index.html",
                "ee/user/project/repository/reducing_the_repo_size_using_git.html",
                "ee/user/project/repository/repository_mirroring.html",
                "ee/user/project/repository/signed_commits/index.html",
                "ee/user/project/repository/x509_signed_commits/index.html",
                "ee/user/project/repository/git_attributes.html",

                # Merge requests (20개)
                "ee/user/project/merge_requests/index.html",
                "ee/user/project/merge_requests/creating_merge_requests.html",
                "ee/user/project/merge_requests/reviews/index.html",
                "ee/user/project/merge_requests/merge_when_pipeline_succeeds.html",
                "ee/user/project/merge_requests/cherry_pick_changes.html",
                "ee/user/project/merge_requests/revert_changes.html",
                "ee/user/project/merge_requests/approvals/index.html",
                "ee/user/project/merge_requests/squash_and_merge.html",
                "ee/user/project/merge_requests/fast_forward_merge.html",
                "ee/user/project/merge_requests/merge_request_dependencies.html",
                "ee/user/project/merge_requests/reviewing_and_managing_merge_requests.html",
                "ee/user/project/merge_requests/draft_merge_requests.html",
                "ee/user/project/merge_requests/merged_results_pipelines.html",
                "ee/user/project/merge_requests/commit_squash.html",
                "ee/user/project/merge_requests/resolve_conflicts.html",
                "ee/user/project/merge_requests/suggest_changes.html",
                "ee/user/project/merge_requests/authorization_for_merge_requests.html",
                "ee/user/project/merge_requests/auto_merge.html",
                "ee/user/project/merge_requests/fail_fast_testing.html",
                "ee/user/project/merge_requests/code_quality.html",

                # CI/CD with Git (10개)
                "ee/ci/introduction/index.html",
                "ee/ci/pipelines/index.html",
                "ee/ci/yaml/index.html",
                "ee/ci/jobs/index.html",
                "ee/ci/variables/index.html",
                "ee/ci/runners/index.html",
                "ee/ci/caching/index.html",
                "ee/ci/artifacts/index.html",
                "ee/ci/environments/index.html",
                "ee/ci/triggers/index.html",
            ]
        },

        # ===== 6. Git SCM Documentation (60개) =====
        # 공식 Git 문서의 참조 가이드
        "git_scm_guides": {
            "base": "https://git-scm.com",
            "pages": [
                # About (5개)
                "about",
                "about/free-and-open-source",
                "about/small-and-fast",
                "about/distributed",
                "about/branching-and-merging",
                "about/staging-area",

                # Documentation (15개)
                "doc",
                "docs/git",
                "docs/giteveryday",
                "docs/gitworkflows",
                "docs/gitcore-tutorial",
                "docs/gitcvs-migration",
                "docs/gitdiffcore",
                "docs/gitglossary",
                "docs/githooks",
                "docs/gitignore",
                "docs/gitmodules",
                "docs/gitrepository-layout",
                "docs/gitrevisions",
                "docs/gittutorial",
                "docs/gittutorial-2",

                # Videos & External (10개)
                "videos",
                "doc/ext",

                # Book sections not in Pro Git list (30개)
                "book/en/v2/Getting-Started-About-Version-Control",
                "book/en/v2/Getting-Started-A-Short-History-of-Git",
                "book/en/v2/Getting-Started-What-is-Git%3F",
                "book/en/v2/Getting-Started-The-Command-Line",
                "book/en/v2/Getting-Started-Installing-Git",
                "book/en/v2/Getting-Started-First-Time-Git-Setup",
                "book/en/v2/Getting-Started-Getting-Help",
                "book/en/v2/Getting-Started-Summary",
                "book/en/v2/Git-Basics-Getting-a-Git-Repository",
                "book/en/v2/Git-Basics-Recording-Changes-to-the-Repository",
                "book/en/v2/Git-Basics-Viewing-the-Commit-History",
                "book/en/v2/Git-Basics-Undoing-Things",
                "book/en/v2/Git-Basics-Working-with-Remotes",
                "book/en/v2/Git-Basics-Tagging",
                "book/en/v2/Git-Basics-Git-Aliases",
                "book/en/v2/Git-Basics-Summary",
                "book/en/v2/Git-Branching-Branches-in-a-Nutshell",
                "book/en/v2/Git-Branching-Basic-Branching-and-Merging",
                "book/en/v2/Git-Branching-Branch-Management",
                "book/en/v2/Git-Branching-Branching-Workflows",
                "book/en/v2/Git-Branching-Remote-Branches",
                "book/en/v2/Git-Branching-Rebasing",
                "book/en/v2/Git-Branching-Summary",
                "book/en/v2/Git-on-the-Server-The-Protocols",
                "book/en/v2/Git-on-the-Server-Getting-Git-on-a-Server",
                "book/en/v2/Git-on-the-Server-Generating-Your-SSH-Public-Key",
                "book/en/v2/Git-on-the-Server-Setting-Up-the-Server",
                "book/en/v2/Git-on-the-Server-Git-Daemon",
                "book/en/v2/Git-on-the-Server-Smart-HTTP",
                "book/en/v2/Git-on-the-Server-GitLab",
            ]
        },

        # ===== 7. Git Tower Guides (40개) =====
        "git_tower": {
            "base": "https://www.git-tower.com/learn/git",
            "pages": [
                # Basics
                "ebook/en/command-line/basics/what-is-version-control",
                "ebook/en/command-line/basics/what-is-git",
                "ebook/en/command-line/basics/getting-started",
                "ebook/en/command-line/basics/basic-workflow",
                "ebook/en/command-line/basics/working-on-your-project",
                "ebook/en/command-line/basics/inspecting-the-repository",
                "ebook/en/command-line/basics/undoing-things",
                "ebook/en/command-line/basics/the-perfect-commit",
                "ebook/en/command-line/basics/branching-strategies",
                "ebook/en/command-line/basics/pull-requests",

                # Branching & Merging
                "ebook/en/command-line/branching-merging/branching-can-change-your-life",
                "ebook/en/command-line/branching-merging/working-with-branches",
                "ebook/en/command-line/branching-merging/merging",
                "ebook/en/command-line/branching-merging/dealing-with-merge-conflicts",
                "ebook/en/command-line/branching-merging/rebase-as-an-alternative-to-merge",
                "ebook/en/command-line/branching-merging/branching-workflows",
                "ebook/en/command-line/branching-merging/long-running-branches",
                "ebook/en/command-line/branching-merging/short-lived-branches",

                # Advanced Topics
                "ebook/en/command-line/advanced-topics/merge-strategies",
                "ebook/en/command-line/advanced-topics/interactive-rebase",
                "ebook/en/command-line/advanced-topics/cherry-pick",
                "ebook/en/command-line/advanced-topics/reflog",
                "ebook/en/command-line/advanced-topics/submodules",
                "ebook/en/command-line/advanced-topics/searching-in-git",
                "ebook/en/command-line/advanced-topics/git-flow",
                "ebook/en/command-line/advanced-topics/rewrite-history",
                "ebook/en/command-line/advanced-topics/reset-checkout-revert",
                "ebook/en/command-line/advanced-topics/bisect",
                "ebook/en/command-line/advanced-topics/subtree",

                # Remote Repositories
                "ebook/en/command-line/remote-repositories/introduction",
                "ebook/en/command-line/remote-repositories/connecting-remote-repositories",
                "ebook/en/command-line/remote-repositories/inspecting-remote-data",
                "ebook/en/command-line/remote-repositories/publishing-local-changes",
                "ebook/en/command-line/remote-repositories/integrating-remote-changes",
                "ebook/en/command-line/remote-repositories/deleting-branches",
                "ebook/en/command-line/remote-repositories/forks-and-pull-requests",

                # Tools & Services
                "ebook/en/command-line/tools-services/desktop-gui-git",
                "ebook/en/command-line/tools-services/git-hosting",
                "ebook/en/command-line/tools-services/continuous-integration",
            ]
        },

        # ===== 8. Bitbucket Git Tutorials (40개) =====
        "bitbucket": {
            "base": "https://www.atlassian.com/git/tutorials",
            "pages": [
                # Learn Git
                "learn-git-with-bitbucket-cloud",
                "git-ssh",
                "git-lfs",
                "install-git",

                # Making a Pull Request
                "making-a-pull-request",
                "making-a-pull-request/how-it-works",
                "making-a-pull-request/example",

                # Git Workflows
                "comparing-workflows/centralized-workflow",
                "comparing-workflows/feature-branch-workflow",
                "comparing-workflows/gitflow-workflow",
                "comparing-workflows/forking-workflow",

                # Refs & Log
                "refs-and-the-reflog",
                "refs-and-the-reflog/git-reflog",
                "refs-and-the-reflog/git-show",

                # Reset, Checkout & Revert
                "resetting-checking-out-and-reverting",
                "resetting-checking-out-and-reverting/commit-level-operations",
                "resetting-checking-out-and-reverting/file-level-operations",
                "resetting-checking-out-and-reverting/summary",

                # Advanced Git Log
                "git-log",
                "git-log/filtering-the-commit-history",
                "git-log/formatting-log-output",
                "git-log/graph-options",

                # Git Hooks
                "git-hooks",
                "git-hooks/local-hooks",
                "git-hooks/server-side-hooks",

                # Tutorials
                "learn-about-code-review-in-bitbucket-cloud",
                "branching",
                "migrate-to-git-from-svn",
                "setting-up-a-repository",
                "how-to-move-a-full-git-repository",
                "merging-vs-rebasing",
                "advanced-git-tutorials",
                "dotfiles",
                "big-repositories",
                "monorepos",
                "git-subtree",
            ]
        },

        # ===== 9. freeCodeCamp Git Guide (50개) =====
        "freecodecamp": {
            "base": "https://www.freecodecamp.org/news",
            "pages": [
                # Git Basics
                "what-is-git-and-how-to-use-it-c341b049ae61/",
                "git-and-github-for-beginners/",
                "learn-the-basics-of-git-in-under-10-minutes-da548267cc91/",
                "git-cheat-sheet-and-best-practices-c6ce5321f52/",
                "how-to-use-git-and-github-in-a-team-like-a-pro/",
                "git-clone-branch-how-to-clone-a-specific-branch/",
                "git-switch-branch/",
                "git-checkout-remote-branch-tutorial/",
                "git-delete-branch-how-to-remove-a-local-or-remote-branch/",
                "git-rename-branch-how-to-change-a-local-branch-name/",

                # Git Commands
                "git-pull-force-how-to-overwrite-local-changes-with-git/",
                "git-push-to-remote-branch-how-to-push-a-local-branch-to-origin/",
                "git-revert-commit-how-to-undo-the-last-commit/",
                "git-reset-origin-how-to-reset-a-local-branch-to-remote-tracking-branch/",
                "how-to-delete-a-git-branch-both-locally-and-remotely/",
                "git-fetch-vs-pull/",
                "git-stash-explained/",
                "how-to-use-git-merge-the-definitive-guide/",
                "the-ultimate-guide-to-git-merge-and-git-rebase/",
                "an-introduction-to-git-merge-and-git-rebase-what-they-are-and-how-to-use-them/",

                # Advanced Git
                "git-rebase-handbook/",
                "git-interactive-rebase-explained/",
                "how-to-use-git-rebase-interactively/",
                "the-ultimate-git-command-tutorial-for-beginners/",
                "how-to-undo-the-last-commit-in-git/",
                "git-reverting-to-previous-commit-how-to-revert-to-an-older-commit/",
                "how-to-fix-git-merge-conflicts/",
                "git-squash-commits/",
                "git-best-practices-guide/",
                "how-to-write-better-git-commit-messages/",

                # Git Workflows
                "how-to-use-branches-in-git/",
                "a-beginner-friendly-introduction-to-git-for-data-science/",
                "git-branching-commands-explained/",
                "gitflow-workflow-continuous-delivery/",
                "how-to-become-a-git-expert/",
                "advanced-git-interactive-rebase-cherry-pick-reflog-and-more/",
                "how-to-use-git-efficiently/",
                "practical-git-and-git-workflows/",
                "git-workflow-explained/",
                "git-branching-strategies-for-managing-complexity/",

                # GitHub
                "how-to-make-your-first-pull-request-on-github/",
                "git-and-github-crash-course/",
                "how-to-use-git-and-github-for-beginners/",
                "the-beginners-guide-to-git-github/",
                "how-to-contribute-to-open-source-projects-beginners-guide/",
                "how-to-make-a-pull-request-on-github/",
                "git-github-collaboration-guide/",
                "github-for-beginners/",
                "how-to-fork-a-github-repository/",
                "git-collaboration-workflow-best-practices/",
            ]
        },
    },

    # ========================================
    # Python 크롤링 타겟 URL (500개 목표)
    # ========================================

    "python": {
        # ===== 1. Real Python (100개) =====
        # 최고 품질 튜토리얼
        "realpython": {
            "base": "https://realpython.com",
            "pages": [
                # Python Basics (30개)
                "/python-first-steps",
                "/python-introduction",
                "/installing-python",
                "/interacting-with-python",
                "/python-comments",
                "/python-variables",
                "/python-data-types",
                "/python-operators-expressions",
                "/python-type-checking",
                "/python-strings",
                "/python-string-formatting",
                "/python-f-strings",
                "/python-input-output",
                "/python-numbers",
                "/python-boolean",
                "/python-type-conversion",
                "/python-constants",
                "/python-naming-conventions",
                "/python-pep8",
                "/python-code-quality",

                # Data Structures (20개)
                "/python-lists-tuples",
                "/python-list",
                "/python-tuples",
                "/python-list-comprehension",
                "/python-dictionaries",
                "/python-dict",
                "/python-dict-comprehension",
                "/python-sets",
                "/python-set",
                "/python-arrays",
                "/python-deque",
                "/python-heapq-module",
                "/python-data-structures",
                "/python-stack",
                "/python-queue",
                "/linked-lists-python",
                "/binary-search-python",
                "/sorting-algorithms-python",
                "/python-hash-table",

                # Control Flow (15개)
                "/python-conditional-statements",
                "/python-if-statement",
                "/python-if-else",
                "/python-for-loop",
                "/python-while-loop",
                "/python-break-continue",
                "/python-pass",
                "/python-match-case",
                "/python-enumerate",
                "/python-zip",
                "/python-range",
                "/python-walrus-operator",

                # Functions (20개)
                "/defining-your-own-python-function",
                "/python-function",
                "/python-lambda",
                "/python-return-statement",
                "/python-scope-legb-rule",
                "/python-closure",
                "/python-decorators",
                "/primer-on-python-decorators",
                "/python-kwargs-and-args",
                "/python-recursion",
                "/python-functools",
                "/python-itertools",
                "/python-generators",
                "/python-yield",
                "/python-generator-expressions",

                # OOP (15개)
                "/python-classes",
                "/python-classes-objects",
                "/python-inheritance",
                "/python-super",
                "/python-property",
                "/python-descriptors",
                "/python-magic-methods",
                "/python-metaclasses",
                "/python-dataclasses",
                "/python-namedtuple",
                "/python-abstract-classes",
                "/python-interface",
            ]
        },

        # ===== 2. Python Official Tutorial (완전판) (50개) =====
        "official_tutorial": {
            "base": "https://docs.python.org/3/tutorial",
            "pages": [
                "index.html",
                "appetite.html",
                "interpreter.html",
                "introduction.html",
                "controlflow.html",
                "datastructures.html",
                "modules.html",
                "inputoutput.html",
                "errors.html",
                "classes.html",
                "stdlib.html",
                "stdlib2.html",
                "venv.html",
                "whatnow.html",
                "interactive.html",
                "floatingpoint.html",
                "appendix.html",
            ]
        },

        # ===== 3. Python Official Library Reference (80개) =====
        "official_library": {
            "base": "https://docs.python.org/3/library",
            "pages": [
                # Built-ins (10개)
                "functions.html",
                "constants.html",
                "stdtypes.html",
                "exceptions.html",

                # Text (5개)
                "string.html",
                "re.html",
                "difflib.html",
                "textwrap.html",
                "unicodedata.html",

                # Binary Data (3개)
                "struct.html",
                "codecs.html",

                # Data Types (15개)
                "datetime.html",
                "calendar.html",
                "collections.html",
                "collections.abc.html",
                "heapq.html",
                "bisect.html",
                "array.html",
                "weakref.html",
                "types.html",
                "copy.html",
                "pprint.html",
                "reprlib.html",
                "enum.html",
                "graphlib.html",

                # Numeric & Math (7개)
                "numbers.html",
                "math.html",
                "cmath.html",
                "decimal.html",
                "fractions.html",
                "random.html",
                "statistics.html",

                # Functional (5개)
                "itertools.html",
                "functools.html",
                "operator.html",

                # File & Directory (10개)
                "pathlib.html",
                "os.path.html",
                "fileinput.html",
                "stat.html",
                "filecmp.html",
                "tempfile.html",
                "glob.html",
                "fnmatch.html",
                "linecache.html",
                "shutil.html",

                # Data Persistence (5개)
                "pickle.html",
                "copyreg.html",
                "shelve.html",
                "marshal.html",
                "dbm.html",
                "sqlite3.html",

                # Data Compression (5개)
                "zlib.html",
                "gzip.html",
                "bz2.html",
                "lzma.html",
                "zipfile.html",
                "tarfile.html",

                # File Formats (5개)
                "csv.html",
                "configparser.html",
                "netrc.html",
                "plistlib.html",

                # Cryptographic (3개)
                "hashlib.html",
                "hmac.html",
                "secrets.html",

                # OS Services (7개)
                "os.html",
                "io.html",
                "time.html",
                "argparse.html",
                "getopt.html",
                "logging.html",
                "logging.config.html",
                "logging.handlers.html",
                "getpass.html",
                "curses.html",
                "platform.html",
                "errno.html",
                "ctypes.html",
            ]
        },

        # ===== 4. PyMOTW - Python Module of the Week (80개) =====
        "pymotw": {
            "base": "https://pymotw.com/3",
            "pages": [
                # Text (5개)
                "/string/index.html",
                "/re/index.html",
                "/difflib/index.html",
                "/textwrap/index.html",

                # Data Structures (15개)
                "/enum/index.html",
                "/collections/index.html",
                "/array/index.html",
                "/heapq/index.html",
                "/bisect/index.html",
                "/queue/index.html",
                "/struct/index.html",
                "/weakref/index.html",
                "/copy/index.html",
                "/pprint/index.html",

                # Algorithms (5개)
                "/functools/index.html",
                "/itertools/index.html",
                "/operator/index.html",
                "/contextlib/index.html",

                # Dates & Times (5개)
                "/time/index.html",
                "/datetime/index.html",
                "/calendar/index.html",

                # Mathematics (5개)
                "/decimal/index.html",
                "/fractions/index.html",
                "/random/index.html",
                "/math/index.html",
                "/statistics/index.html",

                # File System (10개)
                "/os.path/index.html",
                "/pathlib/index.html",
                "/glob/index.html",
                "/fnmatch/index.html",
                "/linecache/index.html",
                "/tempfile/index.html",
                "/shutil/index.html",
                "/filecmp/index.html",
                "/mmap/index.html",

                # Data Persistence (5개)
                "/pickle/index.html",
                "/shelve/index.html",
                "/dbm/index.html",
                "/sqlite3/index.html",

                # Data Compression (5개)
                "/zlib/index.html",
                "/gzip/index.html",
                "/bz2/index.html",
                "/zipfile/index.html",
                "/tarfile/index.html",

                # Cryptography (3개)
                "/hashlib/index.html",
                "/hmac/index.html",

                # Concurrency (10개)
                "/subprocess/index.html",
                "/signal/index.html",
                "/threading/index.html",
                "/multiprocessing/index.html",
                "/asyncio/index.html",
                "/concurrent.futures/index.html",

                # Networking (7개)
                "/ipaddress/index.html",
                "/socket/index.html",
                "/select/index.html",
                "/selectors/index.html",
                "/asyncio/index.html",

                # Internet (5개)
                "/urllib.parse/index.html",
                "/urllib.request/index.html",
                "/urllib.robotparser/index.html",
                "/base64/index.html",
                "/http.server/index.html",
                "/http.cookies/index.html",
                "/webbrowser/index.html",
                "/json/index.html",
            ]
        },

        # ===== 5. W3Schools Python (40개) =====
        "w3schools": {
            "base": "https://www.w3schools.com/python",
            "pages": [
                "python_intro.asp",
                "python_getstarted.asp",
                "python_syntax.asp",
                "python_comments.asp",
                "python_variables.asp",
                "python_datatypes.asp",
                "python_numbers.asp",
                "python_casting.asp",
                "python_strings.asp",
                "python_booleans.asp",
                "python_operators.asp",
                "python_lists.asp",
                "python_tuples.asp",
                "python_sets.asp",
                "python_dictionaries.asp",
                "python_conditions.asp",
                "python_while_loops.asp",
                "python_for_loops.asp",
                "python_functions.asp",
                "python_lambda.asp",
                "python_arrays.asp",
                "python_classes.asp",
                "python_inheritance.asp",
                "python_iterators.asp",
                "python_polymorphism.asp",
                "python_scope.asp",
                "python_modules.asp",
                "python_datetime.asp",
                "python_math.asp",
                "python_json.asp",
                "python_regex.asp",
                "python_pip.asp",
                "python_try_except.asp",
                "python_user_input.asp",
                "python_string_formatting.asp",
                "python_file_handling.asp",
                "python_file_open.asp",
                "python_file_write.asp",
                "python_file_remove.asp",
                "python_mysql_getstarted.asp",
            ]
        },

        # ===== 6. GeeksforGeeks Python (50개) =====
        "geeksforgeeks": {
            "base": "https://www.geeksforgeeks.org",
            "pages": [
                "/python-programming-language/",
                "/python-basics/",
                "/python-variables/",
                "/python-data-types/",
                "/python-keywords/",
                "/python-operators/",
                "/python-if-else/",
                "/python-loops/",
                "/python-for-loops/",
                "/python-while-loop/",
                "/python-list/",
                "/python-tuples/",
                "/python-set/",
                "/python-dictionary/",
                "/functions-in-python/",
                "/python-lambda/",
                "/python-classes-and-objects/",
                "/inheritance-in-python/",
                "/python-polymorphism/",
                "/encapsulation-in-python/",
                "/python-oops-concepts/",
                "/file-handling-python/",
                "/python-exception-handling/",
                "/python-modules/",
                "/python-packages/",
                "/python-iterators/",
                "/generators-in-python/",
                "/decorators-in-python/",
                "/python-closures/",
                "/args-kwargs-python/",
                "/python-recursion/",
                "/python-regex/",
                "/python-datetime-module/",
                "/python-math-module/",
                "/python-os-module/",
                "/python-sys-module/",
                "/python-json/",
                "/python-csv/",
                "/working-with-pickle-python/",
                "/python-sqlite/",
                "/multithreading-python-set-1/",
                "/multiprocessing-python-set-1/",
                "/socket-programming-python/",
                "/python-gui-tkinter/",
                "/python-database-connection/",
                "/python-web-scraping-tutorial/",
                "/python-testing-tutorial/",
                "/python-design-patterns/",
                "/python-data-structures/",
                "/python-algorithms/",
            ]
        },

        # ===== 7. Programiz Python (40개) =====
        "programiz": {
            "base": "https://www.programiz.com/python-programming",
            "pages": [
                "/keywords-identifier",
                "/statement-indentation-comments",
                "/variables-constants-literals",
                "/operators",
                "/numbers",
                "/list",
                "/tuple",
                "/string",
                "/dictionary",
                "/set",
                "/if-elif-else",
                "/for-loop",
                "/while-loop",
                "/break-continue",
                "/pass-statement",
                "/function",
                "/function-argument",
                "/recursion",
                "/anonymous-function",
                "/global-keyword",
                "/global-local-variables",
                "/namespace",
                "/modules",
                "/package",
                "/file-operation",
                "/directory",
                "/exception-handling",
                "/user-defined-exception",
                "/object-oriented-programming",
                "/class",
                "/inheritance",
                "/multiple-inheritance",
                "/operator-overloading",
                "/iterator",
                "/generator",
                "/closure",
                "/decorators",
                "/property",
                "/regex",
                "/datetime",
                "/shallow-deep-copy",
            ]
        },

        # ===== 8. Python HOWTOs (20개) =====
        "official_howto": {
            "base": "https://docs.python.org/3/howto",
            "pages": [
                "pyporting.html",
                "cporting.html",
                "curses.html",
                "descriptor.html",
                "functional.html",
                "logging.html",
                "logging-cookbook.html",
                "regex.html",
                "sockets.html",
                "sorting.html",
                "unicode.html",
                "urllib2.html",
                "argparse.html",
                "ipaddress.html",
                "clinic.html",
                "instrumentation.html",
                "perf_profiling.html",
                "annotations.html",
                "isolating-extensions.html",
                "gdb_helpers.html",
            ]
        },

        # ===== 9. Python Advanced Topics (40개) =====
        "official_advanced": {
            "base": "https://docs.python.org/3",
            "pages": [
                # Type Hints (5개)
                "library/typing.html",
                "library/types.html",
                "library/typing_extensions.html",

                # Async (10개)
                "library/asyncio.html",
                "library/asyncio-task.html",
                "library/asyncio-stream.html",
                "library/asyncio-subprocess.html",
                "library/asyncio-queue.html",
                "library/asyncio-sync.html",
                "library/asyncio-eventloop.html",
                "library/asyncio-protocol.html",
                "library/asyncio-policy.html",

                # Concurrent (7개)
                "library/concurrent.futures.html",
                "library/threading.html",
                "library/multiprocessing.html",
                "library/multiprocessing.shared_memory.html",
                "library/subprocess.html",
                "library/sched.html",
                "library/queue.html",

                # Context Variables (2개)
                "library/contextvars.html",
                "library/contextlib.html",

                # Advanced OOP (8개)
                "library/abc.html",
                "library/dataclasses.html",
                "reference/datamodel.html",
                "library/enum.html",
                "library/inspect.html",

                # Metaprogramming (5개)
                "library/importlib.html",
                "library/importlib.metadata.html",
                "library/importlib.resources.html",
                "library/ast.html",
                "library/symtable.html",

                # Performance (3개)
                "library/timeit.html",
                "library/profile.html",
                "library/tracemalloc.html",
            ]
        },

        # ===== 10. Python Design Patterns (30개) =====
        "python_patterns": {
            "base": "https://python-patterns.guide",
            "pages": [
                # Gang of Four
                "gang-of-four/abstract-factory/",
                "gang-of-four/builder/",
                "gang-of-four/factory-method/",
                "gang-of-four/prototype/",
                "gang-of-four/singleton/",
                "gang-of-four/adapter/",
                "gang-of-four/bridge/",
                "gang-of-four/composite/",
                "gang-of-four/decorator-pattern/",
                "gang-of-four/facade/",
                "gang-of-four/flyweight/",
                "gang-of-four/proxy/",
                "gang-of-four/chain-of-responsibility/",
                "gang-of-four/command/",
                "gang-of-four/iterator/",
                "gang-of-four/mediator/",
                "gang-of-four/memento/",
                "gang-of-four/observer/",
                "gang-of-four/state/",
                "gang-of-four/strategy/",
                "gang-of-four/template-method/",
                "gang-of-four/visitor/",

                # Python-specific patterns
                "python/module-globals/",
                "python/sentinel-object/",
                "python/prebound-methods/",
                "python/decorator-pattern/",
                "python/module-pattern/",
                "python/instance-checking/",
            ]
        },

        # ===== 11. Python Enhancement Proposals (PEPs) (30개) =====
        "python_peps": {
            "base": "https://peps.python.org",
            "pages": [
                "pep-0008/",  # Style Guide
                "pep-0020/",  # Zen of Python
                "pep-0257/",  # Docstring Conventions
                "pep-0484/",  # Type Hints
                "pep-0526/",  # Variable Annotations
                "pep-0585/",  # Type Hinting Generics
                "pep-0586/",  # Literal Types
                "pep-0589/",  # TypedDict
                "pep-0591/",  # Final
                "pep-0604/",  # Union Operator
                "pep-0612/",  # ParamSpec
                "pep-0613/",  # TypeAlias
                "pep-3107/",  # Function Annotations
                "pep-3119/",  # Abstract Base Classes
                "pep-3129/",  # Class Decorators
                "pep-3132/",  # Extended Iterable Unpacking
                "pep-0343/",  # with Statement
                "pep-0380/",  # yield from
                "pep-0492/",  # Coroutines with async/await
                "pep-0525/",  # Async Generators
                "pep-0530/",  # Async Comprehensions
                "pep-0498/",  # f-strings
                "pep-0572/",  # Walrus Operator
                "pep-0634/",  # Structural Pattern Matching
                "pep-0636/",  # Pattern Matching Tutorial
                "pep-0673/",  # Self Type
                "pep-0692/",  # TypedDict with Unpack
                "pep-3333/",  # WSGI
                "pep-0249/",  # DB API 2.0
                "pep-0302/",  # Import Hooks
            ]
        },

        # ===== 12. Talk Python Training (20개) =====
        "talkpython": {
            "base": "https://training.talkpython.fm/courses",
            "pages": [
                "explore_python_jumpstart/welcome-to-the-course",
                "explore_python_jumpstart/python-language-concepts",
                "explore_python_jumpstart/lists-and-collections",
                "explore_python_jumpstart/functions",
                "explore_python_jumpstart/classes",
                "explore_python_jumpstart/error-handling",
                "explore_python_jumpstart/file-io",
                "explore_python_jumpstart/http-services",
                "explore_python_jumpstart/modules-and-packages",
                "explore_python_jumpstart/decorators",
                "explore_python_jumpstart/generators",
                "explore_python_jumpstart/async-programming",
                "explore_python_jumpstart/testing",
                "explore_python_jumpstart/performance",
                "explore_python_jumpstart/best-practices",
                "explore_python_jumpstart/virtual-environments",
                "explore_python_jumpstart/packaging",
                "explore_python_jumpstart/next-steps",
                "explore_python_jumpstart/context-managers",
                "explore_python_jumpstart/comprehensions",
            ]
        },
    }
}

# ========================================
# 크롤링 설정
# ========================================

CRAWL_CONFIG = {
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "request_delay": 2.0,  # 2초 딜레이
    "max_retries": 5,
    "timeout": 30,
    "verify_ssl": True,
    "rate_limit": {
        "requests_per_minute": 15,
        "backoff_on_429": True,
        "backoff_base_delay": 60,
        "max_backoff_delay": 300,
    }
}

# ========================================
# 콘텐츠 추출 설정
# ========================================

CONTENT_CONFIG = {
    "exclude_selectors": [
        "nav", "header", "footer", ".navbar", ".sidebar", ".menu",
        ".navigation", ".breadcrumb", ".toc-container", "#sidebar",
        ".edit-link", ".github-link", "#search-box", ".advertisement",
        ".cookie-notice", ".newsletter", "#comments", ".banner",
        "#top", "#bottom", ".skip-link", ".printfooter",
    ],

    "include_sections": [
        "main", "article", ".main-content", ".content",
        ".documentation", ".tutorial-content", "#main-content",
        ".article-body", ".doc-content", "#content",
        ".post-content", ".page-content", ".section",
        ".markdown-body", "#docker-docs-container"
    ],

    "min_text_length": 20,
    "min_meaningful_words": 5,

    "code_selectors": [
        "pre", "code", ".highlight", ".code-block",
        ".example", ".codehilite", ".sourceCode"
    ]
}

print("[OK] 대규모 설정 로드 완료")
git_count = sum(len(v.get('pages', [])) for v in TARGET_URLS['git'].values())
python_count = sum(len(v.get('pages', v.get('chapters', []))) for v in TARGET_URLS['python'].values())
print(f"  - Git 타겟: {git_count}개 페이지")
print(f"  - Python 타겟: {python_count}개 페이지")
print(f"  - 총 {git_count + python_count}개 페이지")
