#!/usr/bin/python3

# Note: code runs out of memory at 0.025 degree interval

from shine_source_gen import SHINE_Source

cases = {
    'name'        : ['DD'       , 'DT'       , 'DT1pct'   ],
    'T_frac'      : [0.0        , 1.0        , 0.01       ],  # Tritium fraction
    'voltage'     : [280.0      , 280.0      , 280.0      ],  # Beam voltage [kV]
    'current'     : [38.3959    , 38.3959    , 38.3959    ],  # Beam current [mA]
    'pressure'    : [38.397285  , 38.397285  , 38.397285  ],  # Target pressure [torr]
    'temperature' : [1800.0     , 1800.0     , 1800.0     ],  # Target temperature [C]
    'length'      : [137.3025   , 137.3025   , 137.3025   ],  # Target length [cm]
    'start_radius': [0.2        , 0.2        , 0.2        ],  # Beam max radius [cm]
    'max_radius'  : [4.1783     , 4.1783     , 4.1783     ],  # Beam max radius [cm]
    'divergence'  : [36.0       , 36.0       , 36.0       ],  # Beam max radius [cm]
}

def main():

    names = ['DD', 'DT', 'DT1pct']

    for name in names:

        i = cases['name'].index(name)

        T_frac       = cases['T_frac'      ][i]  # Tritium fraction
        voltage      = cases['voltage'     ][i]  # Beam voltage [kV]
        current      = cases['current'     ][i]  # Beam current [mA]
        pressure     = cases['pressure'    ][i]  # Target pressure [torr]
        temperature  = cases['temperature' ][i]  # Target temperature [C]
        length       = cases['length'      ][i]  # Target length [cm]
        start_radius = cases['start_radius'][i]  # Beam start radius [cm]
        max_radius   = cases['max_radius'  ][i]  # Beam max radius [cm]
        divergence   = cases['divergence'  ][i]  # Beam divergence [mrad]
        num_x_ints   = 1370                      # Number of position intervals (internal calculation)
        num_x_ints_C = 137                       # Number of position intervals (output)
        angle_int    = 0.01                      # Angle interval [deg] (internal calculation)
        angle_int_C  = 3.0                       # Angle interval [deg] (output)
        EN_int       = 0.01                      # Neutron energy interval [MeV]

        SS = SHINE_Source(name, T_frac, voltage, current, pressure, temperature, length, start_radius, max_radius,
                          divergence, num_x_ints, num_x_ints_C, angle_int, angle_int_C, EN_int)
        SS.write_hdf5()
        SS.write_mcnp_sdef()
        SS.make_plots()

if __name__ == '__main__': main()
