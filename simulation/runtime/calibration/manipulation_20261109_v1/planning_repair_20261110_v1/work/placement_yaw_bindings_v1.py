from contextlib import contextmanager


@contextmanager
def bind_routes(S, new_routes):
    rebuild = getattr(new_routes, "rebuild", None)
    route = getattr(new_routes, "route", None)
    if not callable(rebuild):
        raise TypeError("new_routes.rebuild must be callable")
    if not callable(route):
        raise TypeError("new_routes.route must be callable")

    M = S.M
    E = S.E
    if not hasattr(M, "routes"):
        raise AttributeError("S.M.routes does not exist")
    if not callable(getattr(E, "rebuild", None)):
        raise TypeError("S.E.rebuild must be callable")

    old_routes = M.routes
    old_rebuild = E.rebuild

    M.routes = new_routes
    E.rebuild = rebuild
    try:
        yield
    finally:
        M.routes = old_routes
        E.rebuild = old_rebuild
