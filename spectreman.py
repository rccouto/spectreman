#!/usr/bin/env python

import math, re, optparse, operator, os, glob
from scipy.interpolate import interp1d
import numpy as np
import sys
from subprocess import call
import time

print ("""
#============================================================#
#                                                            #
#                         Spectreman                         #
#                                                            #
#                  Stockholm, April 13th, 2017               #
#                    Wrote by Rafael C. Couto                #
#                                                            #
#============================================================#
""")


"""
Read the potential energy curves, extend and spline them, and run XAS and RIXS
for the given combinations specified in the input file.

INPUT:

Files:
   Number of gs PEC


"""

# Load data from file
def load(file):
    import numpy
    f = open(file, 'r')
    a = []
    for l in f.readlines():
        try:
            n = list(map(float,l.replace(',',' ').split()))
            if len(n)>0:
                a.append(n)
        except ValueError:
            pass
    f.close()
    return numpy.array(a)


# Read Input
class Input(list):
    def __init__(self): # Class Instantiation
        pass            
    @staticmethod       # Decorator

    def read(): # Read input 'spec.input'
        import numpy as npy
        global init
        global decay
        global final
        global rf
        global np
        global typ
        global espectyp
        global cpus
        with open('spec.input', 'r+') as f:
            for line in f:
                if line.startswith(".INIT"):
                    init=Input()
                    n=next(f, '')
                    for i in range(int(n)):
                        init.append(str(next(f, '')))
                    init=list(map(lambda s: s.strip(), init))
                    #print(n)
                elif line.startswith(".DECAY"):
                    decay=Input()
                    n=next(f, '')
                    for i in range(int(n)):
                        decay.append(next(f, ''))
                    decay=list(map(lambda s: s.strip(), decay))
                    #print(decay)                     
                elif line.startswith(".FINAL"):
                    final=Input()
                    n=next(f, '')
                    for i in range(int(n)):
                        final.append(next(f, ''))
                    final=list(map(lambda s: s.strip(), final))
                    #print(final)
                elif line.startswith(".Rfinal"):
                    rf=float(next(f, ''))
                    #print(rf)
                elif line.startswith(".Points"):
                    np= int(next(f, ''))
                    #print(np)
                elif line.startswith(".TYPE"):
                    typ = str(next(f, '')).strip()
                    #print(typ)
                elif line.startswith(".eSPec"):
                    espectyp=next(f, '').strip()
                    #print(espectyp)
                elif line.startswith(".Cpus"):
                    cpus= int(next(f, ''))
                    #print(cpus) 
            f.close()
        return

# PECs treatment
class PEC(list):
    def __init__(self): # Class Instantiation
        pass            
    @staticmethod       # Decorator

    # Extend the all PECs 
    def extend(pec):
        data = load(pec) # load PEC
        x = data[:,0]
        y = data[:,1]

        stp=x[1]-x[0] # r step
        extrapt=int( (rf-x[-1])/stp ) # N extra points
        
        #File name:
        out = pec.replace(".pot", "-extended.pot"); 
        pot = open(out, 'w')
        
        c=1
        ang2bohr=1.889725989
        for j in range(len(x)+extrapt): # Extending PEC and writing it in file
            if j <= (len(x)-1):
                pot.write("%.2f  %.8f\n" % (x[j]*ang2bohr, y[j]) )
            else:
                pot.write("%.2f  %.8f\n" % ((x[-1]+stp*c)*ang2bohr,y[-1]) )
                c=c+1

    # Spline the potentials
    def spline(pec, k, bott, bottid):
        import numpy as npy
        
        # Extended PEC file name:
        out = pec.replace(".pot", "-extended.pot"); 

        data = load(out) # load PEC
        x = data[:,0]
        y = data[:,1]

        # Spline procedure 
        x_range=npy.array([x[0],x[-1]])
        x_spl=npy.linspace(x_range[0],x_range[1], np)
        spl=npy.zeros((np),dtype=float)
        x2, idx = npy.unique(x, return_index=True)
        y2 = y[idx]
        y_spl = npy.empty(x_spl.size, dtype=float)
        f_spl = interp1d(x2,y2,kind='cubic')
        y_spl = f_spl(x_spl)

        # Splined PEC files
        out = pec.replace(".pot", "-spline.pot"); 
        pot = open(out, 'w')

        # Get the new botton of ground state PEC
        if k == 1:
            bott, bottid = min((bott, bottid) for (bottid, bott) in enumerate(y_spl))
            
        for i in range(np): # Splining PEC and writing it in file
            if i == 0:
                pot.write("# PEC from Spectreman\n")
            pot.write("%.5f  %.10f\n" % (x_spl[i], y_spl[i]-bott) )
            
        if k == 1:
            return bott, bottid
            
# MAIN PROGRAM
def main():
    start_time = time.time()
    
    # Read input file spec.input
    print("\n  .Spectreman input read!")
    Input.read()
    
    # Extend and spline all potentials
    pecs=(init, decay, final)
    if (typ == "extend" or typ == "spline" or typ == "all"):
        k=0
        imin=0
        iminid=0
        for pec in pecs:
            k=k+1
            for i in range(len(pec)):
                
                # Extend
                if (typ == "extend" or typ == "all"):
                    if k == 1:
                        print("  .PECs extended!")
                    PEC.extend(pec[i])
                
                # Spline
                if (typ == "spline" or typ == "all"):
                    if k == 1:
                        print("  .PECs splined!")
                    if k == 1:
                        imin,iminid=PEC.spline(pec[i],k, imin, iminid)
                    else:
                        PEC.spline(pec[i], k, imin, iminid)

    # Run eSPec-Raman
    if (typ == "espec" or typ == "all"):
        print("  .Starting eSPec-Raman procedure:")
        print("    - It will use %s cpus." % cpus)
        fl=0
        for ipec in init:  # Write input file for all combinations of PECs
            ipot = ipec.replace(".pot", "-spline.pot")
            if typ == "espec":
                data = load(ipot) # load PEC
                y = data[:,1]
                imin, iminid = min((imin, iminid) for (iminid, imin) in enumerate(y))
                
            for dpec in decay:
                # Get vertical transition energy and minimum of decay PEC
                dpot = dpec.replace(".pot", "-spline.pot")
                data = load(dpot) # load PEC
                y = data[:,1]
                dmin = min(float(s) for s in y) # Minimum 
                vert=y[iminid] # Vertical transition
                
                for fpec in final:
                    # Get minimum of final PEC
                    fpot = fpec.replace(".pot", "-spline.pot")
                    data = load(fpot) # load PEC
                    y = data[:,1]
                    fmin = min(float(s) for s in y) # Minimum of final PEC
                    
                    # Write input file for eSPec
                    input="{a}-{b}-{c}.ram".format(a=ipec.replace(".pot", ""), b=dpec.replace(".pot", ""), c=fpec.replace(".pot", ""))
                    # Directory name
                    dir=input.replace(".ram", "")
                    
                    # Write eSPec input
                    inp=open(input, 'w')
                    inp.write("""# Raman 1D input
# Spectreman code

model 1d

jobid  %s
dimension .1D
npoints %s
mass 6.857

initial_pot  %s
decaying_pot %s
final_pot    %s

gamma 0.08
step 5D-5
detuning 0.0

Vg_min  %f
Vd_min  %f 
Vf_min  %f
Vd_vert %f

init_time 100.0
fin_time 140.0

shift

window 0.01
print_level full

absorb_cond 0.0007
absorb_range 1.0 11.0
                    """ % (dir, np, ipot, dpot, fpot, imin, dmin, fmin, vert) )
                    inp.close()
                    print("    - Input %s written" % input)
                    
                    # Write script to run eSPec-Raman
                    if fl == 0:
                        fespec=open("run-espec.sh", 'w')
                        fespec.write("#!/bin/bash\n\nsource /opt/intel/compilers_and_libraries_2017.0.098/linux/bin/compilervars.sh intel64\n")

                    fespec.write("""\nmkdir %s
cp %s %s # ipot
cp %s %s # dpot
cp %s %s # fpot
cp %s %s # input

cd %s\n""" % (dir, ipot, dir, dpot, dir, fpot, dir, input, dir, dir))

                    if cpus > 1:
                        fespec.write("/home/rafael/theochem/progs/espec/eSPec-RAMAN/paral_run_espec_raman_floki.sh -%s %s &> log &\n\n" % (espectyp, input))

                
                    else:
                        fespec.write("\necho     Running %s\n\n" % input)
                        
                        fespec.write("/home/rafael/theochem/progs/espec/eSPec-RAMAN/paral_run_espec_raman_floki.sh -%s %s &> log \n\n" % (espectyp, input))
                    fespec.write("cd ../ \n")
                    if (cpus > 1 and fl == (cpus-2)):
                        fespec.write("wait \n")
                        
                    fl = fl + 1
        fespec.close()
        
        # Run the eSPec-Raman script
        print("  .Start to run eSPec-Raman:")
        call("chmod +x run-espec.sh", shell=True)
        call("./run-espec.sh", shell=True)

                    
    print("\n  Spectreman finished!!!\n  Have a nice day!!!\n")
    total=time.time() - start_time
    if total > 60: 
        print("  Total running time:  %.2f minutes\n" % total/60)
    else:
        print("  Total running time:  %.2f seconds\n" % total)
        
if __name__=="__main__":
    main()


