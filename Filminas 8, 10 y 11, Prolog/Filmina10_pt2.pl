% en la quinta actividad me di cuenta que podía hacer todo junto
%me costo entender los conceptos pero no esta tan difícil 
% Estructura general, reglas
oracion(o(SN, SV)) --> sn(SN, Num, Gen), sv(SV, Num).

%Gen es de genero, ósea si es M o F, mejor dicho como masculino y femenino.
%Num es por el numero gramática, ósea si es SG o PL, es decir singular y plural
sn(sn(Det, N), Num, Gen) --> det(Det, Gen, Num), n(N, Gen, Num).

sv(sv(V), Num) --> vi(V, Num).
sv(sv(V, SN), Num) --> vt(V, Num), sn(SN, _, _).
sv(sv(V, Adv), Num) --> vi(V, Num), adv(Adv).
sv(sv(V, N), Num) --> vi(V, Num), n(N, _, _).

% Determinantes con sus géneros y numeros
det(det(los), m, pl) --> [los].
det(det(la), f, sg) --> [la].
det(det(las), f, pl) --> [las].
det(det(el), m, sg) --> [el].
det(det(una), f, sg) --> [una].
det(det(un), m, sg) --> [un].

%Nombres
n(n(aztecas), m, pl) --> [aztecas].
n(n(dominacion), f, sg) --> [dominacion].
n(n(reyes), m, pl) --> [reyes].
n(n(catalan), m, sg) --> [catalan].
n(n(mayas), m, pl) --> [mayas].
n(n(sacerdotes), m, pl) --> [sacerdotes].
n(n(lenguas), f, pl) --> [lenguas].
n(n(espanol), m, sg) --> [espanol].
n(n(terreno), m, sg) --> [terreno].
n(n(ninos), m, pl) --> [ninos].
n(n(granada), f, sg) --> [granada].

% Verbos intransitivos
vi(vi(hablan), pl) --> [hablan].
vi(vi(disminuyo), sg) --> [disminuyo].
vi(vi(resistieron), pl) --> [resistieron].
vi(vi(pasan), pl) --> [pasan].

% Verbos transitivos
vt(vt(conquistaron), pl) --> [conquistaron].
vt(vt(evangelizaban), pl) --> [evangelizaban].
vt(vt(representaba), sg) --> [representaba].

%adverbios
adv(adv(progresivamente)) --> [progresivamente].