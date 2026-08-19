#from .snaptools import *
from ..globaldefs import *
from .. import eos

import numpy as npy
import os
import h5py
import struct


def load_snapshot(snap, fname, headonly=False, thermo=False, compress=False, mats=[402, 400]):
    """
    Loads snapshot data from file
    """
    if not (h5py.is_hdf5(fname) or str(fname).count('.hdf5') > 0):
        load_G2_1(snap, fname, headonly=headonly, thermo=thermo, mats=mats)
    else:
        load_hdf5(snap, fname, headonly=headonly, thermo=thermo)

    # REARRANGE
    snap.pos = npy.array(snap.pos).reshape((snap.N, 3))
    snap.x = snap.pos.T[0]
    snap.y = snap.pos.T[1]
    snap.z = snap.pos.T[2]

    snap.vel = npy.array(snap.vel).reshape((snap.N, 3))
    snap.vx = snap.vel.T[0]
    snap.vy = snap.vel.T[1]
    snap.vz = snap.vel.T[2]

    if snap.N <= 5e9 and compress:
        snap.id = snap.id.astype('uint32', copy=False)
    if compress:
        snap.m = snap.m.astype('float32', copy=False)
        snap.pos = npy.array(snap.pos).astype('float32', copy=False)
        snap.vel = npy.array(snap.vel).astype('float32', copy=False)
        if len(snap.S) > 0:
            snap.S = snap.S.astype('float32', copy=False)
        snap.rho = snap.rho.astype('float32', copy=False)
        snap.hsml = snap.hsml.astype('float32', copy=False)
        snap.pot = snap.pot.astype('float32', copy=False)
        if len(snap.U) > 0:
            snap.U = snap.U.astype('float32', copy=False)
        if thermo:
            if len(snap.P) > 0:
                snap.P = snap.P.astype('float32', copy=False)
            if len(snap.T) > 0:
                snap.T = snap.T.astype('float32', copy=False)
            if len(snap.cs) > 0:
                snap.cs = snap.cs.astype('float32', copy=False)


def load_G2_1(snap, fname, headonly=False, thermo=False, mats=[402, 400]):
    """
    Load a snapshot in Gadget's standard file format (1)
    """
    
    f = open(fname, 'rb')

    struct.unpack('i', f.read(4))  #SKIP

    #HEADER
    snap.header.npart = npy.array(struct.unpack('iiiiii', f.read(24)))
    snap.header.mass = npy.array(struct.unpack('dddddd', f.read(48)))
    (snap.header.time, snap.header.redshift, snap.header.flag_sfr,
        snap.header.flag_feedbacktp) = struct.unpack('ddii', f.read(24))
    snap.header.npartTotal = npy.array(struct.unpack('iiiiii', f.read(24)))
    (snap.header.flag_cooling, snap.header.num_files) = struct.unpack('ii', f.read(8))
    (snap.header.BoxSize,) = struct.unpack('d', f.read(8))
    (snap.header.Omega0, snap.header.OmegaLambda, snap.header.HubbleParam,
        snap.header.flag_stellarage,
        snap.header.flag_metals) = struct.unpack('dddii', f.read(32))
    snap.header.nallhw = npy.array(struct.unpack('iiiiii', f.read(24)))
    (snap.header.flag_entr_ics,) = struct.unpack('i', f.read(4))
    struct.unpack('60x', f.read(60))

    struct.unpack('i', f.read(4))  #SKIP

    if snap.header.num_files != 1:
        print("WARNING! Number of files:", snap.header.num_files,
               ", not currently supported.\n")

    snap.N = snap.header.npart[0]
    snap.file = fname
    snap.inclthermo = thermo
    
    if headonly:
        f.close()
        return

    count = str(snap.N)     # number of particle values to read
    count3 = str(3*snap.N)  # number of values to read for 3-vectors

    #PARTICLE DATA
    struct.unpack('i', f.read(4))  #SKIP
    snap.pos = struct.unpack(count3 + 'f', f.read(3*snap.N*4))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    snap.vel = struct.unpack(count3 + 'f', f.read(3*snap.N*4))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    snap.id = npy.array(struct.unpack(count + 'i', f.read(snap.N*4)))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    snap.m = npy.array(struct.unpack(count + 'f', f.read(snap.N*4)))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    snap.S = npy.array(struct.unpack(count + 'f', f.read(snap.N*4)))
    struct.unpack('i', f.read(4))  #SKIP

    if (snap.S == 0).all():
        snap.S = npy.empty(0)

    struct.unpack('i', f.read(4))  #SKIP
    snap.rho = npy.array(struct.unpack(count + 'f', f.read(snap.N*4)))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    snap.hsml = npy.array(struct.unpack(count + 'f', f.read(snap.N*4)))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    snap.pot = npy.array(struct.unpack(count + 'f', f.read(snap.N*4)))
    struct.unpack('i', f.read(4))  #SKIP

    if thermo:
        if len(f.read(4)) == 4:     #SKIP
            snap.P = npy.array(struct.unpack(count + 'f', f.read(snap.N*4)))
            struct.unpack('i', f.read(4))  #SKIP
        else:
            snap.ensure_matIDs(mats)
            snap.P = eos.calcprop('P', 'rho', 'S', snap.rho, snap.S, snap.materialIDs)
        
        if len(f.read(4)) == 4:     #SKIP
            snap.T = npy.array(struct.unpack(count + 'f', f.read(snap.N*4)))
            struct.unpack('i', f.read(4))  #SKIP
        else:
            snap.ensure_matIDs(mats)
            snap.T = eos.calcprop('T', 'rho', 'S', snap.rho, snap.S, snap.materialIDs)
        
        if len(f.read(4)) == 4:     #SKIP
            snap.U = npy.array(struct.unpack(count+'f', f.read(snap.N*4)))
            struct.unpack('i', f.read(4))  #SKIP
        else:
            snap.ensure_matIDs(mats)
            snap.U = eos.calcprop('U', 'rho', 'S', snap.rho, snap.S, snap.materialIDs)
        
        if len(f.read(4)) == 4:     #SKIP
            snap.cs = npy.array(struct.unpack(count+'f', f.read(snap.N*4)))
            struct.unpack('i', f.read(4))  #SKIP
        
#            if len(f.read(4)) == 4: # acceleration near end in _long format
#                snap.accel = struct.unpack(count3 + 'f', f.read(3*snap.N*4))
#                struct.unpack('i', f.read(4))  #SKIP
        
#            if len(f.read(4)) == 4:
#                snap.dt = npy.array(struct.unpack(count+'f', f.read(snap.N*4)))
#                struct.unpack('i', f.read(4))  #SKIP
        
    #if len(snap.accel) > 0:
    #   snap.accel = npy.array(snap.accel).reshape((snap.N, 3))
    #   snap.ax = snap.accel.T[0]
    #   snap.ay = snap.accel.T[1]
    #   snap.az = snap.accel.T[2]

    #print("Read", snap.N, "particles.\n")
    f.close()
    
    if os.path.exists(str(snap.file)+'_rem.txt'):
        ids, rems = npy.loadtxt(str(snap.file)+'_rem.txt', unpack=True)
        if npy.array_equal(ids, snap.id):
            snap.rem = rems
        else:
            print('array mismatch')


def load_hdf5(snap, fname, headonly=False, recenter=True, thermo=False, debug=False):
    """
    Load an HDF5 snapshot
    """
    
    with h5py.File(fname, 'r') as f:
        header = f.get('Header')
        part = f['PartType0']
        #tid = npy.where(f['PartType0/ParticleIDs'][:] < PROJ_ID_OFFSET)[0]
        #pid = npy.where(f['PartType0/ParticleIDs'][:] >= PROJ_ID_OFFSET)[0]

        if 'Units' in f.keys():
            units = f.get('Units')
            if debug:
                print('setting conversion factors')
            Lfactor = units.attrs["Unit length in cgs (U_L)"]
            Mfactor = units.attrs["Unit mass in cgs (U_M)"]
            Tfactor = units.attrs["Unit time in cgs (U_t)"]
            if npy.ndim(Lfactor) > 0:
                Lfactor = Lfactor[0]
            if npy.ndim(Mfactor) > 0:
                Mfactor = Mfactor[0]
            if npy.ndim(Tfactor) > 0:
                Tfactor = Tfactor[0]
        else:
            Lfactor = Mfactor = Tfactor = 1.

        snap.header.npart = header.attrs['NumPart_ThisFile']
        snap.header.mass = header.attrs['MassTable'] * Mfactor
        snap.header.time = header.attrs['Time'] * Tfactor
        if npy.ndim(snap.header.time) > 0:
            snap.header.time = snap.header.time[0]
        snap.header.redshift = 0.0
        snap.header.flag_sfr = 0
        snap.header.flag_feedbacktp = 0
        snap.header.npartTotal = header.attrs['NumPart_Total']
        snap.header.flag_cooling = 0
        if npy.ndim(header.attrs['NumFilesPerSnapshot']) > 0:
            snap.header.num_files = header.attrs['NumFilesPerSnapshot'].max()
        else:
            snap.header.num_files = header.attrs['NumFilesPerSnapshot']
        if npy.ndim(header.attrs['BoxSize']) > 0:
            snap.header.BoxSize = (header.attrs['BoxSize']).max() * Lfactor
        else:
            snap.header.BoxSize = header.attrs['BoxSize'] * Lfactor
        snap.header.Omega0 = 0.0
        snap.header.OmegaLambda = 0.0
        snap.header.HubbleParam = 1.0
        snap.header.flag_stellarage = 0
        snap.header.flag_metals = 0
        snap.header.nallhw = npy.zeros(6).astype(int)
        if npy.ndim(header.attrs['Flag_Entropy_ICs']) > 0:
            snap.header.flag_entr_ics = header.attrs['Flag_Entropy_ICs'][0]
        else:
            snap.header.flag_entr_ics = header.attrs['Flag_Entropy_ICs']

        snap.N = snap.header.npart[0]
        snap.file = fname
        snap.inclthermo = thermo
    
        if headonly:
            print('only load header')
            f.close()
            return
    
        snap.pos = part['Coordinates'][:].reshape((snap.header.npart[0], 3)) * Lfactor
        if recenter:
            snap.pos -= snap.header.BoxSize/2.
        snap.vel = part['Velocities'][:].reshape((snap.header.npart[0], 3)) * Lfactor/Tfactor
        if 'MaterialIDs' in part.keys():
            snap.materialIDs = part['MaterialIDs'][:]
#### edit
        # move to conversion routines?
        if part['ParticleIDs'][:].max() < GADGET_EOS_OFFSET and len(npy.unique(snap.materialIDs)) > 1:
            snap.id = npy.where(snap.materialIDs > 400, part['ParticleIDs'][:], part['ParticleIDs'][:]+GADGET_EOS_OFFSET)
            snap.id = npy.where(snap.materialIDs < 400, snap.id+GADGET_EOS_OFFSET, snap.id)
#### edit end
        else:
            snap.id = part['ParticleIDs'][:]
        snap.m = part['Masses'][:] * Mfactor
        snap.rho = part['Densities'][:] * Mfactor/(Lfactor**3)
        snap.hsml = part['SmoothingLengths'][:] * Lfactor
        snap.U = part['InternalEnergies'][:] * Lfactor**2/(Tfactor**2)
        if 'Entropies' in part.keys() and part['Entropies'][:].max() > 0:
            snap.S = part['Entropies'][:] * Lfactor**2/(Tfactor**2)
        elif thermo:
            snap.S = eos.calcprop('S', 'U', 'rho', snap.U, snap.rho, snap.materialIDs)
        if 'Pressures' in part.keys():
            snap.P = part['Pressures'][:] * Mfactor / (Lfactor * Tfactor**2)
        elif thermo:
            snap.P = eos.calcprop('P', 'U', 'rho', snap.U, snap.rho, snap.materialIDs)
        if 'Temperatures' in part.keys():
            snap.T = part['Temperatures'][:]
        elif thermo:
            snap.T = eos.calcprop('T', 'U', 'rho', snap.U, snap.rho, snap.materialIDs)
        if 'Potentials' in part.keys():
            snap.pot = part['Potentials'][:] * Lfactor**2/(Tfactor**2)
        if 'RemnantIDs' in part.keys():
            snap.rem = part['RemnantIDs'][:]

    if debug:
        print("Read", snap.N, "particles.\n")


def load_seagen(snap, partplanet, thermo=False, init_h=100e5):
    """
    Assign snapshot particle data from seagen particleplanet
    """
    #HEADER
    snap.header.npart = npy.array([partplanet.N_picle, 0, 0, 0, 0, 0])
    snap.header.mass = npy.array([0., 0., 0., 0., 0., 0.])
    snap.header.time = 0.
    snap.header.redshift = 0.
    snap.header.flag_sfr = snap.header.flag_feedbacktp = snap.header.flag_cooling = 0
    snap.header.npartTotal = snap.header.npart
    snap.header.num_files = 1
    snap.header.BoxSize = 0.0
    snap.header.Omega0 = snap.header.OmegaLambda = 0.0
    snap.header.HubbleParam = 1.0
    snap.header.flag_stellarage = snap.header.flag_metals = 0
    snap.header.nallhw = npy.array([0, 0, 0, 0, 0, 0])
    snap.header.flag_entr_ics = 1

    snap.N = snap.header.npart[0]

    #PARTICLE DATA
    snap.x = partplanet.x
    snap.y = partplanet.y
    snap.z = partplanet.z
    snap.pos = npy.array((snap.x, snap.y, snap.z))
    snap.pos = snap.pos.T
    
    snap.vx = npy.zeros(snap.N)
    snap.vy = npy.zeros(snap.N)
    snap.vz = npy.zeros(snap.N)
    snap.vel = npy.zeros((snap.N, 3))

    extraIDoff = [len(partplanet.mat[partplanet.mat < x]) for x in npy.unique(partplanet.mat)]
    extraIDoff = npy.array(extraIDoff)
    snap.id = npy.arange(len(partplanet.mat)) + partplanet.mat * (GADGET_EOS_OFFSET) - extraIDoff[partplanet.mat]

    snap.m = partplanet.m
    snap.S = partplanet.S
    snap.rho = partplanet.rho
    snap.P = partplanet.P
    snap.T = partplanet.T
    ##snap.U = eos.calcprop('U', 'rho', 'S', snap.rho, snap.S, snap.materialIDs) # handled elsewhere
    
    snap.hsml = npy.ones(snap.N) * init_h
    snap.pot = npy.zeros(snap.N)


def write_snapshot(snap, fname):
    """
    Write snapshot to file
    """
    if not (h5py.is_hdf5(fname) or str(fname).count('.hdf5') > 0):
        write_G2_1(snap, fname)
    else:
        write_hdf5(snap, fname)


def write_G2_1(snap, fname):

    f = open(fname, 'wb')

    f.write(struct.pack('i', 256))  #SKIP

    #HEADER
    if len(snap.header.npart > 6):
        npart = snap.header.npart[0:6]
        mass = snap.header.mass[0:6]
        npartTotal = snap.header.npartTotal[0:6]
    else:
        npart = snap.header.npart
        mass = snap.header.mass
        npartTotal = snap.header.npartTotal
    f.write(struct.pack('iiiiii', *npart))
    f.write(struct.pack('dddddd', *mass))
    f.write(struct.pack('ddii', snap.header.time, snap.header.redshift,
                 snap.header.flag_sfr, snap.header.flag_feedbacktp))
    f.write(struct.pack('iiiiii', *npartTotal))
    f.write(struct.pack('iiddddii', snap.header.flag_cooling,
                 snap.header.num_files, snap.header.BoxSize,
                 snap.header.Omega0, snap.header.OmegaLambda,
                 snap.header.HubbleParam, snap.header.flag_stellarage,
                 snap.header.flag_metals))
    f.write(struct.pack('iiiiii', *snap.header.nallhw))
    f.write(struct.pack('i', snap.header.flag_entr_ics))
    f.write(struct.pack('60x'))

    f.write(struct.pack('i', 256))  #SKIP

    if snap.header.num_files != 1:
        print("WARNING! Number of files:", snap.header.num_files,
               ", not currently supported.\n")

    count = str(snap.N)
    count3 = str(3*snap.N)

    #PARTICLE DATA
    f.write(struct.pack('i', snap.N))  #SKIP
    f.write(struct.pack(count3 + 'f', *npy.array(snap.pos).reshape((3*snap.N))))
    f.write(struct.pack('i', snap.N))  #SKIP

    f.write(struct.pack('i', snap.N))  #SKIP
    f.write(struct.pack(count3 + 'f', *npy.array(snap.vel).reshape((3*snap.N))))
    f.write(struct.pack('i', snap.N))  #SKIP

    f.write(struct.pack('i', snap.N))  #SKIP
    f.write(struct.pack(count + 'i', *snap.id))
    f.write(struct.pack('i', snap.N))  #SKIP

    f.write(struct.pack('i', snap.N))  #SKIP
    f.write(struct.pack(count + 'f', *snap.m))
    f.write(struct.pack('i', snap.N))  #SKIP

    f.write(struct.pack('i', snap.N))  #SKIP
    if len(snap.S) == len(snap.id):
        f.write(struct.pack(count + 'f', *snap.S))
    else:
        f.write(struct.pack(count + 'f', *npy.zeros(snap.N)))
    f.write(struct.pack('i', snap.N))  #SKIP

    f.write(struct.pack('i', snap.N))  #SKIP
    f.write(struct.pack(count + 'f', *snap.rho))
    f.write(struct.pack('i', snap.N))  #SKIP

    f.write(struct.pack('i', snap.N))  #SKIP
    f.write(struct.pack(count + 'f', *snap.hsml))
    f.write(struct.pack('i', snap.N))  #SKIP
    
    f.write(struct.pack('i', snap.N))  #SKIP
    f.write(struct.pack(count + 'f', *snap.pot))
    f.write(struct.pack('i', snap.N))  #SKIP

    print("Wrote", snap.N, "particles.\n")
    f.close()


def write_hdf5(snap, outname, units='cgs', mats=[401, 400], shift2center=True):

    snap.G2_to_swift(mats=mats, fname=outname, write=False)
    
    if npy.ndim(snap.header.flag_entr_ics) < 1:
        if snap.header.flag_entr_ics == 1:
            #print('Entropy in U block')
            intEblock = snap.S
        else:   ## normal for converting gadget2-planetary to swift
            intEblock = snap.U
    else:
        if snap.header.flag_entr_ics[0] == 1:
            print('Entropy in U block. Warning! not expected for Swift!')
            intEblock = snap.S
        else:   ## normal for converting gadget2-planetary to swift
            intEblock = snap.U
    
    if units == 'SI':
        Lfactor = eos.uconversion_l_cgs2SI
        Mfactor = eos.uconversion_m_cgs2SI
        Tfactor = 1.
    else:
        Lfactor = Mfactor = Tfactor = 1.
    
    with h5py.File(outname, 'w') as f:
        # SnapHeader
        header = f.create_group("/Header")
        header.attrs['NumPart_ThisFile'] = snap.header.npart
        header.attrs['MassTable'] = snap.header.mass * Mfactor
        header.attrs['Time'] = snap.header.time * Tfactor
        header.attrs['Redshift'] = 0.0
        header.attrs['Flag_Sfr'] = 0
        header.attrs['Flag_Feedback'] = 0
        header.attrs['NumPart_Total'] = snap.header.npart
        header.attrs['Flag_Cooling'] = 0
        header.attrs['NumFilesPerSnapshot'] = snap.header.num_files
        header.attrs['BoxSize'] = [snap.header.BoxSize * Lfactor, snap.header.BoxSize * Lfactor, snap.header.BoxSize * Lfactor]
        header.attrs['Omega0'] = 0.0
        header.attrs['OmegaLambda'] = 0.0
        header.attrs['HubbleParam'] = 1.0
        header.attrs['Flag_StellarAge'] = 0
        header.attrs['Flag_Metals'] = 0
        header.attrs['NumPart_Total_HighWord'] = npy.zeros(6).astype(int)
        header.attrs['Flag_Entropy_ICs'] = snap.header.flag_entr_ics
        
        # Units
        units = f.create_group('Units')
        units.attrs["Unit length in cgs (U_L)"] = 1./Lfactor
        units.attrs["Unit mass in cgs (U_M)"] = 1./Mfactor
        units.attrs["Unit time in cgs (U_t)"] = 1./Tfactor
        units.attrs["Unit current in cgs (U_I)"] = 1.0
        units.attrs["Unit temperature in cgs (U_T)"] = 1.0
        
        # Particles
        part = f.create_group('/PartType0/')
        if shift2center:
            part.create_dataset('Coordinates', data=(snap.pos.ravel() + snap.header.BoxSize/2.) * Lfactor, compression='gzip')
        else:
            part.create_dataset('Coordinates', data=(snap.pos.ravel()) * Lfactor, compression='gzip')
        part.create_dataset('Velocities', data=snap.vel.ravel() * Lfactor/Tfactor, compression='gzip')
        part.create_dataset('MaterialIDs', data=snap.materialIDs, compression='gzip')
        part.create_dataset('ParticleIDs', data=snap.id, compression='gzip')
        part.create_dataset('Masses', data=snap.m * Mfactor, compression='gzip')
        f['/PartType0/Internal Energy'] = part.create_dataset('InternalEnergies', data=intEblock * Lfactor**2/(Tfactor**2), compression='gzip')
        f['/PartType0/Density'] = part.create_dataset('Densities', data=snap.rho * Mfactor/Lfactor**3, compression='gzip')
        f['/PartType0/SmoothingLength'] = part.create_dataset('SmoothingLengths', data=snap.hsml * Lfactor, compression='gzip')
        f['/PartType0/Potential'] = part.create_dataset('Potentials', data=snap.pot * Lfactor**2/(Tfactor**2), compression='gzip')
        if len(snap.S) == len(snap.id):
            part.create_dataset('Entropies', data=npy.where(npy.isnan(snap.S), 0.0, snap.S * Lfactor**2/(Tfactor**2)), compression='gzip')
        if snap.inclthermo:
            if len(snap.P) == len(snap.id):
                part.create_dataset('Pressures', data=snap.P * Mfactor / (Lfactor * Tfactor**2), compression='gzip')
            if len(snap.T) == len(snap.id):
                part.create_dataset('Temperatures', data=snap.T, compression='gzip')
        if snap.rem:
            part.create_dataset('RemnantIDs', data=snap.rem, compression='gzip')


def save_remnant_ids(snap):
    if h5py.is_hdf5(snap.file):
        with h5py.File(snap.file, 'a') as f:
            part = f['PartType0']
            if 'RemnantIDs' in part.keys():
                part['RemnantIDs'][:] = snap.rem
            else:
                part.create_dataset('RemnantIDs', data=snap.rem, compression='gzip')
    else:
        npy.savetxt(snap.file+'_rem.txt', npy.transpose([snap.id, snap.rem]), header='Id  Remnant', fmt='%d')
