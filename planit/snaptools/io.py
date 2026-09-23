#from .snaptools import *
from ..globaldefs import *
from .. import eos

import numpy as npy
import os
import h5py
import struct

# tipsy data types
tipsy_header_type = npy.dtype([('time', '>f8'), ('N', '>u4'), ('Dims', '>u4'), ('Ngas', '>u4'), ('Ndark', '>u4'), ('Nstar', '>u4'), ('pad', '>u4')])
dark_type = npy.dtype([('mass','>f4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('vx', '>f4'), ('vy', '>f4'), ('vz', '>f4'), ('eps','>f4'), ('phi','>f4')])
gas_type  = npy.dtype([('mass','>f4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('vx', '>f4'), ('vy', '>f4'), ('vz', '>f4'), ('rho','>f4'), ('temp','>f4'), ('hsmooth','>f4'), ('metals','>f4'), ('phi','>f4')])


def load_snapshot(snap, fname, headonly=False, thermo=False, compress=False, mats=[402, 400], loadprops=['all',]):
    """
    Loads snapshot data from file
    """
    if headonly:
        loadprops = ['header',]
    
    if not (h5py.is_hdf5(fname) or str(fname).count('.hdf5') > 0):
        with open(fname, 'rb') as f:
            i = struct.unpack('i', f.read(4))
        if i[0] == 256: # this could break if the first 4 bytes of time in a tipsy file give thes same int
            load_G2_1(snap, fname, headonly=headonly, thermo=thermo, mats=mats, loadprops=loadprops)
        else:
            load_tipsy(snap, fname, headonly=headonly, thermo=thermo, loadprops=loadprops)
    else:
        load_hdf5(snap, fname, headonly=headonly, thermo=thermo, loadprops=loadprops)

    if not headonly:
        # REARRANGE
        if any(x in ['all','x','y','z'] for x in loadprops):
            snap.pos = npy.array(snap.pos).reshape((snap.N, 3))
        if any(x in ['all','x'] for x in loadprops):
            snap.x = snap.pos.T[0]
        if any(x in ['all','y'] for x in loadprops):
            snap.y = snap.pos.T[1]
        if any(x in ['all','z'] for x in loadprops):
            snap.z = snap.pos.T[2]
    
        if any(x in ['all','vx','vy','vz'] for x in loadprops):
            snap.vel = npy.array(snap.vel).reshape((snap.N, 3))
        if any(x in ['all','vx'] for x in loadprops):
            snap.vx = snap.vel.T[0]
        if any(x in ['all','vy'] for x in loadprops):
            snap.vy = snap.vel.T[1]
        if any(x in ['all','vz'] for x in loadprops):
            snap.vz = snap.vel.T[2]
    
        if snap.N <= 5e9 and any(x in ['all','id'] for x in loadprops) and compress:
            snap.id = snap.id.astype('uint32', copy=False)
        if compress:
            if any(x in ['all','m'] for x in loadprops):
                snap.m = snap.m.astype('float32', copy=False)
            if any(x in ['all','x','y','z'] for x in loadprops):
                snap.pos = npy.array(snap.pos).astype('float32', copy=False)
            if any(x in ['all','vx','vy','vz'] for x in loadprops):
                snap.vel = npy.array(snap.vel).astype('float32', copy=False)
            if any(x in ['all','S'] for x in loadprops) and len(snap.S) > 0:
                snap.S = snap.S.astype('float32', copy=False)
            if any(x in ['all','rho'] for x in loadprops):
                snap.rho = snap.rho.astype('float32', copy=False)
            if any(x in ['all','hsml'] for x in loadprops):
                snap.hsml = snap.hsml.astype('float32', copy=False)
            if any(x in ['all','pot'] for x in loadprops):
                snap.pot = snap.pot.astype('float32', copy=False)
            if any(x in ['all','U'] for x in loadprops) and len(snap.U) > 0:
                snap.U = snap.U.astype('float32', copy=False)
            if thermo:
                if any(x in ['all','P'] for x in loadprops) and len(snap.P) > 0:
                    snap.P = snap.P.astype('float32', copy=False)
                if any(x in ['all','T'] for x in loadprops) and len(snap.T) > 0:
                    snap.T = snap.T.astype('float32', copy=False)
                if any(x in ['all','cs'] for x in loadprops) and len(snap.cs) > 0:
                    snap.cs = snap.cs.astype('float32', copy=False)


def load_G2_1(snap, fname, headonly=False, thermo=False, mats=[402, 400], loadprops=['all',]):
    """
    Load a snapshot in Gadget's standard file format (1)
    """
    
    f = open(fname, 'rb')

    struct.unpack('i', f.read(4))  #SKIP

    #HEADER
    if any(x in ['all','header'] for x in loadprops):
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
    else:
        f.read(260)
        
    if headonly:
        f.close()
        return

    count = str(snap.N)     # number of particle values to read
    count3 = str(3*snap.N)  # number of values to read for 3-vectors

    #PARTICLE DATA
    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(3*snap.N*4)
    if any(x in ['all','x','y','z'] for x in loadprops):
        snap.pos = struct.unpack(count3 + 'f', buffer)
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(3*snap.N*4)
    if any(x in ['all','vx','vy','vz'] for x in loadprops):
        snap.vel = struct.unpack(count3 + 'f', buffer)
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(snap.N*4)
    if any(x in ['all','id'] for x in loadprops):
        snap.id = npy.array(struct.unpack(count + 'i', buffer))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(snap.N*4)
    if any(x in ['all','m'] for x in loadprops):
        snap.m = npy.array(struct.unpack(count + 'f', buffer))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(snap.N*4)
    if any(x in ['all','S'] for x in loadprops):
        snap.S = npy.array(struct.unpack(count + 'f', buffer))
        if (snap.S == 0).all():
            snap.S = npy.empty(0)
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(snap.N*4)
    if any(x in ['all','rho'] for x in loadprops):
        snap.rho = npy.array(struct.unpack(count + 'f', buffer))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(snap.N*4)
    if any(x in ['all','hsml'] for x in loadprops):
        snap.hsml = npy.array(struct.unpack(count + 'f', buffer))
    struct.unpack('i', f.read(4))  #SKIP

    struct.unpack('i', f.read(4))  #SKIP
    buffer = f.read(snap.N*4)
    if any(x in ['all','pot'] for x in loadprops):
        snap.pot = npy.array(struct.unpack(count + 'f', buffer))
    struct.unpack('i', f.read(4))  #SKIP

    if thermo:
        if len(f.read(4)) == 4:     #SKIP
            buffer = f.read(snap.N*4)
            if any(x in ['all','P'] for x in loadprops):
                snap.P = npy.array(struct.unpack(count + 'f', buffer))
            struct.unpack('i', f.read(4))  #SKIP
        elif any(x in ['all','P'] for x in loadprops):
            snap.ensure_matIDs(mats)
            snap.P = eos.calcprop('P', 'rho', 'S', snap.rho, snap.S, snap.materialIDs)
        
        if len(f.read(4)) == 4:     #SKIP
            buffer = f.read(snap.N*4)
            if any(x in ['all','T'] for x in loadprops):
                snap.T = npy.array(struct.unpack(count + 'f', buffer))
            struct.unpack('i', f.read(4))  #SKIP
        elif any(x in ['all','T'] for x in loadprops):
            snap.ensure_matIDs(mats)
            snap.T = eos.calcprop('T', 'rho', 'S', snap.rho, snap.S, snap.materialIDs)
        
        if len(f.read(4)) == 4:     #SKIP
            buffer = f.read(snap.N*4)
            if any(x in ['all','U'] for x in loadprops):
                snap.U = npy.array(struct.unpack(count+'f', buffer))
            struct.unpack('i', f.read(4))  #SKIP
        elif any(x in ['all','U'] for x in loadprops):
            snap.ensure_matIDs(mats)
            snap.U = eos.calcprop('U', 'rho', 'S', snap.rho, snap.S, snap.materialIDs)
        
        if len(f.read(4)) == 4:     #SKIP
            buffer = f.read(snap.N*4)
            if any(x in ['all','cs'] for x in loadprops):
                snap.cs = npy.array(struct.unpack(count+'f', buffer))
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
    
    if os.path.exists(str(snap.file)+'_rem.txt') and any(x in ['all','rem','bnd'] for x in loadprops):
        ids, rems = npy.loadtxt(str(snap.file)+'_rem.txt', unpack=True)
        if npy.array_equal(ids, snap.id):
            snap.rem = rems
        else:
            print('array mismatch')


def load_hdf5(snap, fname, headonly=False, recenter=True, thermo=False, debug=False, loadprops=['all',]):
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

        if any(x in ['all','header'] for x in loadprops):
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
            f.close()
            return
    
        if any(x in ['all','x','y','z'] for x in loadprops):
            snap.pos = part['Coordinates'][:].reshape((snap.header.npart[0], 3)) * Lfactor
            if recenter:
                snap.pos -= snap.header.BoxSize/2.
        if any(x in ['all','vx','vy','vz'] for x in loadprops):
            snap.vel = part['Velocities'][:].reshape((snap.header.npart[0], 3)) * Lfactor/Tfactor
        if 'MaterialIDs' in part.keys() and any(x in ['all','materialIDs'] for x in loadprops):
            snap.materialIDs = part['MaterialIDs'][:]
        if any(x in ['all','id'] for x in loadprops):
#### edit
            # move to conversion routines?
            if part['ParticleIDs'][:].max() < GADGET_EOS_OFFSET and len(npy.unique(snap.materialIDs)) > 1:
                snap.id = npy.where(snap.materialIDs > 400, part['ParticleIDs'][:], part['ParticleIDs'][:]+GADGET_EOS_OFFSET)
                snap.id = npy.where(snap.materialIDs < 400, snap.id+GADGET_EOS_OFFSET, snap.id)
#### edit end
            else:
                snap.id = part['ParticleIDs'][:]
        if any(x in ['all','m'] for x in loadprops):
            snap.m = part['Masses'][:] * Mfactor
        if any(x in ['all','rho'] for x in loadprops):
            snap.rho = part['Densities'][:] * Mfactor/(Lfactor**3)
        if any(x in ['all','hsml'] for x in loadprops):
            snap.hsml = part['SmoothingLengths'][:] * Lfactor
        if any(x in ['all','U'] for x in loadprops):
            snap.U = part['InternalEnergies'][:] * Lfactor**2/(Tfactor**2)
        if 'Entropies' in part.keys() and part['Entropies'][:].max() > 0 and any(x in ['all','S'] for x in loadprops):
            snap.S = part['Entropies'][:] * Lfactor**2/(Tfactor**2)
        elif thermo and any(x in ['all','S'] for x in loadprops):
            snap.S = eos.calcprop('S', 'U', 'rho', snap.U, snap.rho, snap.materialIDs)
        if 'Pressures' in part.keys() and any(x in ['all','P'] for x in loadprops):
            snap.P = part['Pressures'][:] * Mfactor / (Lfactor * Tfactor**2)
        elif thermo and any(x in ['all','P'] for x in loadprops):
            snap.P = eos.calcprop('P', 'U', 'rho', snap.U, snap.rho, snap.materialIDs)
        if 'Temperatures' in part.keys() and any(x in ['all','T'] for x in loadprops):
            snap.T = part['Temperatures'][:]
        elif thermo and any(x in ['all','T'] for x in loadprops):
            snap.T = eos.calcprop('T', 'U', 'rho', snap.U, snap.rho, snap.materialIDs)
        if 'Potentials' in part.keys() and any(x in ['all','pot'] for x in loadprops):
            snap.pot = part['Potentials'][:] * Lfactor**2/(Tfactor**2)
        if 'RemnantIDs' in part.keys() and any(x in ['all','rem','bnd'] for x in loadprops):
            snap.rem = part['RemnantIDs'][:]

    if debug:
        print("Read", snap.N, "particles.\n")


def decode_tipsy_header(header):
    """
       Decode TIPSY file header particle numbering.
       Written by Thomas Meier
    """
    pad  = int(header['pad'])
    N    = int(header['N'])     + ((pad&0x000000ff)<<32)
    nGas = int(header['Ngas'])  + ((pad&0x0000ff00)<<24)
    nDark= int(header['Ndark']) + ((pad&0x00ff0000)<<16)
    nStar= int(header['Nstar']) + ((pad&0xff000000)<< 8)
    if nGas + nDark + nStar != N:
        N = int(header['N'])
        nGas = int(header['Ngas'])
        nDark= int(header['Ndark'])
        nStar= int(header['Nstar'])
    print(f'Total: {N}, Gas:{nGas}, Dark:{nDark}, Star:{nStar}')
    return N, nGas, nDark, nStar


def encode_tipsy_header(N, Ngas, Ndark, Nstar):
    """
        Encode full 64-bit particle counts into:
        32-bit low parts (stored individually)
        8-bit high parts packed into pad
        
        Written by Thomas Meier
    """
    # low 32-bit parts
    N_low = N & 0xffffffff
    Ngas_low = Ngas & 0xffffffff
    Ndark_low = Ndark & 0xffffffff
    Nstar_low = Nstar & 0xffffffff
    # high 8-bit parts (bits 32–39 of each value)
    N_high = (N >> 32) & 0xff
    Ngas_high = (Ngas >> 32) & 0xff
    Ndark_high = (Ndark >> 32) & 0xff
    Nstar_high = (Nstar >> 32) & 0xff
    # pack into 32-bit pad
    pad = (
    (N_high << 0) |
    (Ngas_high << 8) |
    (Ndark_high << 16) |
    (Nstar_high << 24)
    )
    
    return N_low, Ngas_low, Ndark_low, Nstar_low, pad


def load_tipsy(snap, fname, headonly=False, recenter=False, thermo=False, debug=False, loadprops=['all',]):
    """
       Load a TIPSY format snapshot.
       Adapted from code provided by Thomas Meier
    """
 
    # UNITS
    # G=1, kpc in code units, solar mass in code units. Length unit is 1 RE, v is 1 km/s, G = 1
    # M = 1/62.5476 M_Earth
    # RHO = 0.368411779571 g/cm^3
    # T = 1 R_Earth / 1 km/s = 6378 s = 1.772 h
    Lfactor = Rearth
    Mfactor = 1/62.5476 * Mearth
    Tfactor = Rearth/1.e5 # (Rearth/(1 km/s))
   
    with open(fname,'rb') as tipsy:
        tipsyheader = npy.fromfile(tipsy, dtype=tipsy_header_type, count=1)
        N, nGas, nDark, nStar = decode_tipsy_header(tipsyheader[0])
        if not headonly:
            if nDark>0:
                dark = npy.fromfile(tipsy, dtype=dark_type, count=nDark)
            gas  = npy.fromfile(tipsy, dtype=gas_type, count=nGas)

    snap.header.npart = npy.array([tipsyheader['N'][0], 0, 0, 0, 0, 0])
    snap.header.mass = npy.array([0., 0., 0., 0., 0., 0.])
    snap.header.time = tipsyheader['time'] * Tfactor
    snap.header.redshift = 0.
    snap.header.flag_sfr = snap.header.flag_feedbacktp = snap.header.flag_cooling = 0
    snap.header.npartTotal = snap.header.npart
    snap.header.num_files = 1
    snap.header.BoxSize = 0.0
    snap.header.Omega0 = snap.header.OmegaLambda = 0.0
    snap.header.HubbleParam = 1.0
    snap.header.flag_stellarage = snap.header.flag_metals = 0
    snap.header.nallhw = npy.array([0, 0, 0, 0, 0, 0])
    snap.header.flag_entr_ics = 0

    snap.N = snap.header.npart[0]
    
    snap.file = fname
    snap.inclthermo = thermo
    
    if headonly:
        return
    
    
    #PARTICLE DATA
    snap.x = gas['x'] * Lfactor
    snap.y = gas['y'] * Lfactor
    snap.z = gas['z'] * Lfactor
    snap.pos = npy.array((snap.x, snap.y, snap.z))
    snap.pos = snap.pos.T
    
    snap.vx = gas['vx'] * Lfactor/Tfactor
    snap.vy = gas['vy'] * Lfactor/Tfactor
    snap.vz = gas['vz'] * Lfactor/Tfactor
    snap.vel = npy.array((snap.vx, snap.vy, snap.vz))
    snap.vel = snap.vel.T
    
    # set up particle IDs to be consistent with Gadget numbering (if possible)
    # particle order is consistent so could switch proj id when material changes back
    if snap.N < GADGET_EOS_OFFSET:
        extraIDoff = [len(gas['metals'][gas['metals'] == x]) for x in npy.unique(gas['metals'])]
        extraIDoff = npy.append(npy.array(0),npy.array(extraIDoff))
        materialint = npy.unique(gas['metals'], return_inverse=True)[1]
        snap.id = npy.arange(len(gas['metals'])) + materialint * (GADGET_EOS_OFFSET) - extraIDoff[materialint]
    else:
        snap.id = npy.arange(len(gas['metals']))
        print('Warning: particle count exceeds Gadget2 particle ID limit')

    snap.m = gas['mass'].astype(float) * Mfactor
    snap.rho = gas['rho'].astype(float) * Mfactor/(Lfactor**3)
    snap.T = gas['temp'].astype(float)
    snap.materialIDs = eos.pkdgrav3towoma(gas['metals'])
    
    if any(x in ['all','S'] for x in loadprops):
        snap.S = eos.calcprop('S', 'rho', 'T', snap.rho, snap.T, snap.materialIDs)
    if any(x in ['all','P'] for x in loadprops):
        snap.P = eos.calcprop('P', 'rho', 'T', snap.rho, snap.T, snap.materialIDs)
    if any(x in ['all','U'] for x in loadprops):
        snap.U = eos.calcprop('U', 'rho', 'T', snap.rho, snap.T, snap.materialIDs)
    
    snap.hsml = gas['hsmooth'] * Lfactor
    snap.pot = gas['phi'] * Lfactor**2/(Tfactor**2)
    
    # load remnant data if it exists
    if os.path.exists(str(snap.file)+'_rem.txt') and any(x in ['all','rem','bnd'] for x in loadprops):
        ids, rems = npy.loadtxt(str(snap.file)+'_rem.txt', unpack=True)
        if npy.array_equal(ids, snap.id):
            snap.rem = rems
        else:
            print('array mismatch')


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
        # if flagged as having entropy ICs assume is Gadget2 format, otherwise tipsy
        if snap.header.flag_entr_ics != 1 or str(fname).count('.std') > 0:
            write_tipsy(snap, fname)
        else:
            write_G2_1(snap,fname)
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


def write_tipsy(snap, outname, units='default', mats=[401, 400]):
    """
       Write a TIPSY format snapshot.
       Adapted from code provided by Thomas Meier
       Conversion from Gadget2 not yet implemented.
    """
 
    # UNITS
    # G=1, kpc in code units, solar mass in code units. Length unit is 1 RE, v is 1 km/s, G = 1
    # M = 1/62.5476 M_Earth
    # RHO = 0.368411779571 g/cm^3
    # T = 1 R_Earth / 1 km/s = 6378 s = 1.772 h
    Lfactor = 1./(Rearth)
    Mfactor = 1./(1/62.5476 * Mearth)
    Tfactor = 1./(Rearth/1.e5) # ((1 km/s)/Rearth)


    N_low, Ngas_low, Ndark_low, Nstar_low, pad = encode_tipsy_header(snap.header.npart[0],snap.header.npart[0],0,0)

    header = npy.zeros(1,dtype=tipsy_header_type)
    header['time'] = snap.header.time * Tfactor
    header['N'] = N_low
    header['Dims'] = 3
    header['Ngas'] = Ngas_low
    header['Ndark'] = Ndark_low
    header['Nstar'] = Nstar_low
    header['pad'] = pad

    gas = npy.zeros((snap.header.npart[0],), dtype=gas_type)
    #dark = npy.zeros((N_dark,), dtype=dark_type)
    
    if str(outname)[-4:] != '.std':
        outname = str(outname) + '.std'

    # sort. particle order is consistent so could switch proj id when material changes back
    
    #PARTICLE DATA
    gas['x'] = snap.x * Lfactor
    gas['y'] = snap.y * Lfactor
    gas['z'] = snap.z * Lfactor
    
    gas['vx'] = snap.vx * Lfactor/Tfactor
    gas['vy'] = snap.vy * Lfactor/Tfactor
    gas['vz'] = snap.vz * Lfactor/Tfactor
    
    gas['mass'] = snap.m * Mfactor
    gas['rho'] = snap.rho * Mfactor/(Lfactor**3)
    gas['temp'] = snap.T
    gas['metals'] = eos.womatopkdgrav3(snap.materialIDs)
        
    gas['hsmooth'] = snap.hsml * Lfactor
    gas['phi'] = snap.pot * Lfactor**2/(Tfactor**2)

        
    with open(outname,'wb') as newfile:
        newfile.write(npy.array(header,dtype=tipsy_header_type).tobytes())
        newfile.write(npy.array(gas,dtype=gas_type).tobytes())
        #newfile.write(npy.array(dark,dtype=dark_type).tobytes())
               

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
