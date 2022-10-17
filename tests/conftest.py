from types import ModuleType
from typing import Callable

import pytest

COMPAT_OPTIONS: set[str] = set()


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--compat", nargs="+", choices=list(COMPAT_OPTIONS), default=list()
    )


@pytest.fixture
def compat(request: pytest.FixtureRequest) -> list[str]:
    return request.config.getoption("--compat")


def register_compat_option(name: str, module: str) -> Callable[[set[str]], ModuleType]:
    COMPAT_OPTIONS.add(name)

    @pytest.fixture
    @pytest.mark.usefixtures(compat.__name__)
    def compat_module(compat: set[str]) -> ModuleType:
        return __import__(module) if name in compat else pytest.importorskip(module)

    setattr(compat_module, "__name__", name)

    return compat_module
