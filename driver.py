#!/usr/bin/python3

# Note: code runs out of memory at 0.025 degree interval

from source import SHINE_Source, get_atom_density

cases = {
    'name'       : ['DD'       , 'DT'       , 'DT1pct'   ],
    'T_frac'     : [0.0        , 1.0        , 0.01       ],  # Tritium fraction
    'voltage'    : [280.0      , 280.0      , 280.0      ],  # Beam voltage [kV]
    'current'    : [38.2915    , 38.2915    , 38.2915    ],  # Beam current [mA]
    'pressure'   : [38.397285  , 38.397285  , 38.397285  ],  # Target pressure [torr]
    'temperature': [1800.0     , 1800.0     , 1800.0     ],  # Target temperature [C]
}

def main():

    names = ['DD', 'DT', 'DT1pct']

    for name in names:

        i = cases['name'].index(name)

        T_frac      = cases['T_frac'     ][i]  # Tritium fraction
        voltage     = cases['voltage'    ][i]  # Beam voltage [kV]
        current     = cases['current'    ][i]  # Beam current [mA]
        pressure    = cases['pressure'   ][i]  # Target pressure [torr]
        temperature = cases['temperature'][i]  # Target temperature [C]
        length      = 137.3025                 # Target length [cm]
        num_x_ints  = 1370                     # Number of position intervals
        angle_int   = 0.01                     # Angle interval [deg]
        EN_int      = 0.01                     # Neutron energy interval [MeV]

        adens  = get_atom_density(pressure, temperature)
        SS     = SHINE_Source(voltage, adens, length, num_x_ints, angle_int, EN_int)
        nyield = (SS.nyield['DT'] * T_frac + SS.nyield['DD'] * (1 - T_frac)) * current / 1000
        print('Total neutron yield = %.5e n/s' % (nyield))
        SS.make_plots(name, T_frac, current)

if __name__ == '__main__': main()
