program Calls;

function Helper(X: Integer): Integer;
begin
  Helper := X;
end;

function Caller(X: Integer): Integer;
begin
  Caller := Helper(X);
end;

begin
end.
