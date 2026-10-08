-module(calls).
-export([caller/1]).

helper(X) ->
    X.

caller(X) ->
    helper(X).
