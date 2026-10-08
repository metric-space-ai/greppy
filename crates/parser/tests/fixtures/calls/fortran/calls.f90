function helper(x)
  integer :: helper, x
  helper = x
end function helper

function caller(x)
  integer :: caller, x
  caller = helper(x)
end function caller
