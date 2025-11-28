"""
RAG 크롤러 설정 - Git(180문서) + Python(272문서)
URL 태깅 포함
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
PYTHON_RAW_DIR = RAW_DATA_DIR / "raw" / "python"

# 디렉토리 자동 생성
for dir_path in [RAW_DATA_DIR, GIT_RAW_DIR, PYTHON_RAW_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================================
# 크롤링 타겟 URL (Git + Python만)
# ========================================

TARGET_URLS = {
    "git": {
        # 1. Atlassian Git Tutorial (60문서)
        "atlassian": {
            "base": "https://www.atlassian.com/git/tutorials",
            "pages": [
                # 기초
                "what-is-version-control",
                "what-is-git",
                "install-git",
                # 저장소
                "setting-up-a-repository",
                "setting-up-a-repository/git-init",
                "setting-up-a-repository/git-clone",
                "setting-up-a-repository/git-config",
                # 변경사항 저장
                "saving-changes",
                "saving-changes/git-add",
                "saving-changes/git-commit",
                "saving-changes/git-diff",
                "saving-changes/git-stash",
                "saving-changes/gitignore",
                # 저장소 검사
                "inspecting-a-repository",
                "inspecting-a-repository/git-status",
                "inspecting-a-repository/git-tag",
                "inspecting-a-repository/git-blame",
                "inspecting-a-repository/git-log",
                # 변경사항 되돌리기
                "undoing-changes",
                "undoing-changes/git-clean",
                "undoing-changes/git-revert",
                "undoing-changes/git-reset",
                "undoing-changes/git-rm",
                # 히스토리 재작성
                "rewriting-history",
                "rewriting-history/git-commit--amend",
                "rewriting-history/git-rebase",
                "rewriting-history/git-reflog",
                # 브랜치
                "using-branches",
                "using-branches/git-branch",
                "using-branches/git-checkout",
                "using-branches/git-merge",
                "using-branches/merge-conflicts",
                "using-branches/merge-strategy",
                # 동기화
                "syncing",
                "syncing/git-remote",
                "syncing/git-fetch",
                "syncing/git-push",
                "syncing/git-pull",
                # 워크플로우
                "comparing-workflows",
                "comparing-workflows/centralized-workflow",
                "comparing-workflows/feature-branch-workflow",
                "comparing-workflows/gitflow-workflow",
                "comparing-workflows/forking-workflow",
                # 고급
                "advanced-overview",
                "merging-vs-rebasing",
                "resetting-checking-out-and-reverting",
                "git-hooks",
                "refs-and-the-reflog",
                "git-lfs"
            ]
        },

        # 2. Git 공식 Book (Pro Git) - 한국어 (50문서)
        "pro_git_ko": {
            "base": "https://git-scm.com/book/ko/v2",
            "pages": [
                # 1장: 시작하기
                "/시작하기-버전-관리란%3F",
                "/시작하기-Git-기초",
                "/시작하기-Git-설치",
                "/시작하기-Git-최초-설정",
                "/시작하기-도움말-보기",
                # 2장: Git의 기초
                "/Git의-기초-Git-저장소-만들기",
                "/Git의-기초-수정하고-저장소에-저장하기",
                "/Git의-기초-커밋-히스토리-조회하기",
                "/Git의-기초-되돌리기",
                "/Git의-기초-리모트-저장소",
                "/Git의-기초-태그",
                "/Git의-기초-Git-Alias",
                # 3장: Git 브랜치
                "/Git-브랜치-브랜치란-무엇인가",
                "/Git-브랜치-브랜치와-Merge-의-기초",
                "/Git-브랜치-브랜치-관리",
                "/Git-브랜치-브랜치-워크플로",
                "/Git-브랜치-리모트-브랜치",
                "/Git-브랜치-Rebase-하기",
                # 5장: 분산 환경에서의 Git
                "/분산-환경에서의-Git-분산-환경에서의-Workflow",
                "/분산-환경에서의-Git-프로젝트에-기여하기",
                "/분산-환경에서의-Git-프로젝트-관리하기",
                # 6장: GitHub
                "/GitHub-계정-만들고-설정하기",
                "/GitHub-GitHub-프로젝트에-기여하기",
                "/GitHub-GitHub-프로젝트-관리하기",
                # 7장: Git 도구
                "/Git-도구-리비전-조회하기",
                "/Git-도구-대화형-명령",
                "/Git-도구-Stashing과-Cleaning",
                "/Git-도구-내-작업에-서명하기",
                "/Git-도구-검색",
                "/Git-도구-히스토리-단장하기",
                "/Git-도구-Reset-명확히-알고-가기",
                "/Git-도구-고급-Merge",
                # 10장: Git의 내부
                "/Git의-내부-Plumbing-명령과-Porcelain-명령",
                "/Git의-내부-Git-개체",
                "/Git의-내부-Git-Refs"
            ]
        },

        # 3. GitHub Docs (18문서)
        "github_docs": {
            "base": "https://docs.github.com/en/get-started",
            "pages": [
                # Getting Started with Git
                "getting-started-with-git/set-up-git",
                "getting-started-with-git/configuring-git-to-handle-line-endings",
                "getting-started-with-git/about-remote-repositories",
                "getting-started-with-git/managing-remote-repositories",
                "getting-started-with-git/caching-your-github-credentials-in-git",
                "getting-started-with-git/associating-text-editors-with-git",
                # Using Git
                "using-git/about-git",
                "using-git/pushing-commits-to-a-remote-repository",
                "using-git/getting-changes-from-a-remote-repository",
                "using-git/dealing-with-non-fast-forward-errors",
                "using-git/splitting-a-subfolder-out-into-a-new-repository",
                "using-git/about-git-subtree-merges",
                # Importing Projects
                "importing-your-projects-to-github/importing-source-code-to-github/about-github-importer",
                "importing-your-projects-to-github/importing-source-code-to-github/importing-a-repository-with-github-importer",
                "importing-your-projects-to-github/importing-source-code-to-github/updating-commit-author-attribution-with-github-importer",
                # Writing on GitHub
                "writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax",
                "writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github"
            ]
        },

        # 4. W3Schools (17문서)
        "w3schools": {
            "base": "https://www.w3schools.com/git",
            "pages": [
                # 기초
                "default.asp",
                "git_intro.asp",
                "git_install.asp",
                # 저장소
                "git_new_files.asp",
                "git_staging_environment.asp",
                "git_commit.asp",
                "git_help.asp",
                # 브랜치
                "git_branch.asp",
                "git_branch_merge.asp",
                # 원격
                "git_remote_getstarted.asp",
                "git_remote_send_pull_request.asp",
                # 고급
                "git_clone.asp",
                "git_undo.asp",
                "git_amend.asp",
                "git_revert.asp",
                "git_reset.asp",
                "git_ignore.asp",
                "git_security_ssh.asp"
            ]
        },

        # 5. Git 공식 Reference Manual (35문서 - 주요 명령어만)
        "official_reference": {
            "base": "https://git-scm.com/docs",
            "pages": [
                # 주요 Porcelain Commands
                "git-add",
                "git-branch",
                "git-checkout",
                "git-cherry-pick",
                "git-clean",
                "git-clone",
                "git-commit",
                "git-diff",
                "git-fetch",
                "git-init",
                "git-log",
                "git-merge",
                "git-pull",
                "git-push",
                "git-rebase",
                "git-reset",
                "git-revert",
                "git-rm",
                "git-show",
                "git-stash",
                "git-status",
                "git-tag",
                # 고급 명령어
                "git-bisect",
                "git-submodule",
                "git-worktree",
                "git-reflog",
                "git-gc",
                "git-grep",
                "git-restore",
                "git-switch",
                # 주요 Plumbing Commands
                "git-cat-file",
                "git-hash-object",
                "git-rev-parse",
                "git-show-ref"
            ]
        }
    },

    "python": {
        # 1. Real Python (45문서)
        "realpython": {
            "base": "https://realpython.com",
            "pages": [
                # 기초
                "/python-first-steps",
                "/python-introduction",
                "/installing-python",
                "/interacting-with-python",
                "/python-comments",
                "/python-variables",
                "/python-data-types",
                "/python-operators-expressions",
                "/python-type-checking",
                # 문자열
                "/python-strings",
                "/python-string-formatting",
                "/python-f-strings",
                "/python-input-output",
                # 컬렉션
                "/python-lists-tuples",
                "/python-list-comprehension",
                "/python-dictionaries",
                "/python-dict-comprehension",
                "/python-sets",
                # 제어 흐름
                "/python-conditional-statements",
                "/python-if-else",
                "/python-for-loop",
                "/python-while-loop",
                "/python-break-continue",
                # 함수
                "/defining-your-own-python-function",
                "/python-lambda",
                "/python-return-statement",
                "/python-scope-legb-rule",
                "/python-closure",
                "/python-decorators",
                # 모듈과 패키지
                "/python-modules-packages",
                "/python-import",
                "/python-main-function",
                "/absolute-vs-relative-python-imports",
                # 파일 처리
                "/working-with-files-in-python",
                "/read-write-files-python",
                "/python-pathlib",
                # 예외 처리
                "/python-exceptions",
                "/python-raise-exception",
                "/python-try-except",
                # 객체 지향
                "/python-classes",
                "/python-inheritance",
                "/python-super",
                "/python-property",
                "/python-magic-methods"
            ]
        },

        # 2. Python 공식 튜토리얼 (15문서)
        "official_tutorial": {
            "base": "https://docs.python.org/3/tutorial",
            "chapters": [
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
                "floatingpoint.html"
            ]
        },

        # 3. W3Schools Python (27문서)
        "w3schools": {
            "base": "https://www.w3schools.com/python",
            "pages": [
                # 기초
                "python_intro.asp",
                "python_getstarted.asp",
                "python_syntax.asp",
                "python_comments.asp",
                "python_variables.asp",
                "python_datatypes.asp",
                "python_numbers.asp",
                "python_casting.asp",
                # 문자열
                "python_strings.asp",
                "python_strings_slicing.asp",
                "python_strings_methods.asp",
                "python_string_formatting.asp",
                # 연산자
                "python_operators.asp",
                "python_booleans.asp",
                # 컬렉션
                "python_lists.asp",
                "python_lists_comprehension.asp",
                "python_tuples.asp",
                "python_sets.asp",
                "python_dictionaries.asp",
                # 제어 흐름
                "python_conditions.asp",
                "python_while_loops.asp",
                "python_for_loops.asp",
                # 함수
                "python_functions.asp",
                "python_lambda.asp",
                # 객체 지향
                "python_classes.asp",
                "python_inheritance.asp"
            ]
        },

        # 4. Python Docs - Library Reference (28문서 - 주요 모듈만)
        "official_library": {
            "base": "https://docs.python.org/3/library",
            "pages": [
                # Built-in
                "functions.html",
                "constants.html",
                "stdtypes.html",
                # Text Processing
                "string.html",
                "re.html",
                # Data Types
                "datetime.html",
                "collections.html",
                "array.html",
                "enum.html",
                # Numeric & Math
                "numbers.html",
                "math.html",
                "random.html",
                "statistics.html",
                # Functional Programming
                "itertools.html",
                "functools.html",
                "operator.html",
                # File & Directory
                "pathlib.html",
                "os.path.html",
                "io.html",
                "json.html",
                # Data Persistence
                "pickle.html",
                "sqlite3.html",
                # File Formats
                "csv.html",
                # Generic Operating System
                "os.html",
                "sys.html",
                "argparse.html",
                "logging.html"
            ]
        },

        # 5. Python Official HOWTOs (18문서)
        "official_howto": {
            "base": "https://docs.python.org/3/howto",
            "pages": [
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
                "annotations.html",
                "descriptor.html",
                "pyporting.html",
                "cporting.html",
                "curses.html",
                "clinic.html",
                "instrumentation.html",
                "perf_profiling.html"
            ]
        },

        # 6. GeeksforGeeks Python (30문서 - 핵심만)
        "geeksforgeeks": {
            "base": "https://www.geeksforgeeks.org",
            "pages": [
                "/python-programming-language/",
                "/python-variables/",
                "/python-data-types/",
                "/python-keywords/",
                "/python-operators/",
                "/python-if-else/",
                "/python-loops/",
                "/python-for-loops/",
                "/python-while-loop/",
                "/break-continue-and-pass-in-python/",
                "/python-list/",
                "/python-tuples/",
                "/python-set/",
                "/python-dictionary/",
                "/functions-in-python/",
                "/python-lambda/",
                "/python-closures/",
                "/decorators-in-python/",
                "/args-kwargs-python/",
                "/python-recursion/",
                "/python-oops-concepts/",
                "/python-classes-and-objects/",
                "/inheritance-in-python/",
                "/python-polymorphism/",
                "/file-handling-python/",
                "/python-exception-handling/",
                "/python-modules/",
                "/python-iterators/",
                "/generators-in-python/",
                "/python-magic-methods/"
            ]
        },

        # 7. Programiz Python (26문서)
        "programiz": {
            "base": "https://www.programiz.com/python-programming",
            "pages": [
                # Introduction
                "/keywords-identifier",
                "/statement-indentation-comments",
                "/variables-constants-literals",
                "/operators",
                # Data Types
                "/numbers",
                "/list",
                "/tuple",
                "/string",
                "/dictionary",
                # Flow Control
                "/if-elif-else",
                "/for-loop",
                "/while-loop",
                "/break-continue",
                "/pass-statement",
                # Functions
                "/function",
                "/function-argument",
                "/recursion",
                "/anonymous-function",
                "/global-keyword",
                "/namespace",
                # Files
                "/file-operation",
                "/directory",
                "/exception-handling",
                "/user-defined-exception",
                # OOP
                "/object-oriented-programming",
                "/class"
            ]
        },

        # 8. PyMOTW (45문서 - 주요 모듈)
        "pymotw": {
            "base": "https://pymotw.com/3",
            "pages": [
                # Text Processing
                "/string/index.html",
                "/re/index.html",
                "/difflib/index.html",
                # Data Structures
                "/collections/index.html",
                "/array/index.html",
                "/heapq/index.html",
                "/queue/index.html",
                "/enum/index.html",
                # Algorithms
                "/functools/index.html",
                "/itertools/index.html",
                "/operator/index.html",
                # Dates & Times
                "/datetime/index.html",
                "/time/index.html",
                # Mathematics
                "/math/index.html",
                "/random/index.html",
                "/statistics/index.html",
                # File & Directory
                "/os.path/index.html",
                "/pathlib/index.html",
                "/glob/index.html",
                "/tempfile/index.html",
                "/shutil/index.html",
                # Data Formats
                "/csv/index.html",
                "/json/index.html",
                "/configparser/index.html",
                # Compression
                "/gzip/index.html",
                "/zipfile/index.html",
                # Cryptography
                "/hashlib/index.html",
                # Concurrency
                "/subprocess/index.html",
                "/threading/index.html",
                "/multiprocessing/index.html",
                "/asyncio/index.html",
                # Networking
                "/socket/index.html",
                "/select/index.html",
                "/http.server/index.html",
                # Application Building
                "/argparse/index.html",
                "/logging/index.html",
                "/getpass/index.html",
                "/cmd/index.html",
                # Debugging & Testing
                "/pdb/index.html",
                "/traceback/index.html",
                "/unittest/index.html",
                # Runtime Features
                "/sys/index.html",
                "/os/index.html",
                "/platform/index.html",
                "/gc/index.html"
            ]
        },

        # 9. Python Advanced (38문서)
        "official_advanced": {
            "base": "https://docs.python.org/3",
            "pages": [
                # Type Hints
                "library/typing.html",
                "library/types.html",
                # Async/Await
                "library/asyncio.html",
                "library/asyncio-task.html",
                "library/asyncio-stream.html",
                "library/asyncio-subprocess.html",
                "library/asyncio-queue.html",
                "library/asyncio-sync.html",
                # Concurrent Execution
                "library/concurrent.futures.html",
                "library/threading.html",
                "library/multiprocessing.html",
                "library/subprocess.html",
                # Context Variables
                "library/contextvars.html",
                # Advanced OOP
                "library/abc.html",
                "library/dataclasses.html",
                "reference/datamodel.html",
                # Metaprogramming
                "library/inspect.html",
                "library/importlib.html",
                "library/ast.html",
                # Performance
                "library/timeit.html",
                "library/profile.html",
                "library/tracemalloc.html"
            ]
        }
    }
}

# ========================================
# 크롤링 설정
# ========================================

CRAWL_CONFIG = {
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "request_delay": 2.0,
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

print("[OK] 설정 로드 완료")
print(f"  - Git 타겟: {sum(len(v.get('pages', [])) for v in TARGET_URLS['git'].values())}개 페이지")
print(f"  - Python 타겟: {sum(len(v.get('pages', v.get('chapters', []))) for v in TARGET_URLS['python'].values())}개 페이지")
