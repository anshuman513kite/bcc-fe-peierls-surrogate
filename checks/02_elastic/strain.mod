# One strained state. Needs: ${mode} (x or xy), ${e} (signed strain), ${tag}.
# Starts fresh from the relaxed 250-atom cell of checks/01_lattice.
clear
units           metal
boundary        p p p
atom_style      atomic
read_data       ../01_lattice/relaxed.data
change_box      all triclinic
pair_style      eam/fs
pair_coeff      * * ../../potentials/Fe_mm.eam.fs Fe
thermo_style    custom step pe pxx pyy pzz pxy pxz pyz
thermo_modify   format float %.10g

variable        L0 equal lx
variable        s  equal 1.0+${e}
variable        t  equal ${e}*ly
if "${mode} == x"  then "change_box all x scale ${s} remap units box"
# Shear: tilt xy = e*ly, so the ENGINEERING shear strain gamma_xy = e (tensor eps_xy = e/2)
if "${mode} == xy" then "change_box all xy final ${t} remap units box"

run             0
variable        E1 equal pe
variable        Ebefore equal ${E1}
min_style       cg
minimize        1e-12 1e-12 10000 100000
variable        dE equal pe-${Ebefore}
variable        fmax equal fmax
print           "STATE ${tag} e=${e} lx=$(lx:%.10f) ly=$(ly:%.10f) xy=$(xy:%.10f) dE_min=$(v_dE:%.3e) eV fmax=$(v_fmax:%.3e) pxx=$(pxx:%.6f) pyy=$(pyy:%.6f) pzz=$(pzz:%.6f) pxy=$(pxy:%.6f) bar"
