module Calls where

helper x =
  x

caller y =
  helper (map helper)
