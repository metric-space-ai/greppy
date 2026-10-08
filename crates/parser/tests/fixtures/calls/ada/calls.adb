function Helper (X : Integer) return Integer is
begin
   return X;
end Helper;

function Caller (X : Integer) return Integer is
begin
   return Helper(X);
end Caller;
