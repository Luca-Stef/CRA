mkdir -p CRA/$1/$2
rsync --remove-source-files -z ir-stef1@login-icelake.hpc.cam.ac.uk:"/home/ir-stef1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-stef1/Fe_perfect_AM04/vary_stresses/$1/CRA/$2/*/*.lmp /home/ir-stef1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-stef1/Fe_perfect_AM04/vary_stresses/$1/CRA/$2/*/*.lmp.gz /home/ir-stef1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-stef1/Fe_perfect_AM04/vary_stresses/$1/CRA/$2/2/*.dat" CRA/$1/$2/
echo Transferred $1 $2
