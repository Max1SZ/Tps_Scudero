% Gramática de empleados
oracion(o(SN, SV)) --> sn(SN, Num, Gen), sv(SV, Num).

sn(sn(Det, N), Num, Gen) --> det(Det, Gen, Num), n(N, Gen, Num).
sv(sv(V), Num) --> v(V, Num).
sv(sv(V, SN), Num) --> v(V, Num), sn(SN, _, _).

det(det(la), f, sg) --> [la].
det(det(el), m, sg) --> [el].
det(det(los), m, pl) --> [los].
det(det(las), f, pl) --> [las].

n(n(empleado), m, sg) --> [empleado].
n(n(empleada), f, sg) --> [empleada].
n(n(empleados), m, pl) --> [empleados].
n(n(empleadas), f, pl) --> [empleadas].
n(n(sueldo), m, sg) --> [sueldo].
n(n(sueldos), m, pl) --> [sueldos].

v(v(trabaja), sg) --> [trabaja].
v(v(trabajan), pl) --> [trabajan].
v(v(cobra), sg) --> [cobra].
v(v(cobran), pl) --> [cobran].
