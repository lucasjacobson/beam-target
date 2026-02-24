#!/usr/bin/python3

from source import SHINE_Source, get_atom_density

def main():

    # Note: code runs out of memory at 0.025 degree interval

    T_frac      = 0.9        # Tritium fraction
    voltage     = 280.0      # Beam voltage [kV]
    current     = 38.2915    # Beam current [mA]
    pressure    = 38.397285  # Target pressure [torr]
    temperature = 1800.0     # Target temperature [C]
    length      = 137.3025   # Target length [cm]
    num_x_ints  = 1370       # Number of position intervals
    angle_int   = 0.01       # Angle interval [deg]
    EN_int      = 0.01       # Neutron energy interval

    adens       = get_atom_density(pressure, temperature)
    args        = (T_frac, voltage, current, adens, length, num_x_ints, angle_int, EN_int)

    n_DD = SHINE_Source('DD', *args)
    n_DT = SHINE_Source('DT', *args)

    nyield = n_DD.nyield + n_DT.nyield
    print('Total neutron yield = %.5e n/s' % (nyield))

if __name__ == '__main__': main()
