import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams
from ase.io import read
from pymatgen.io.vasp import Poscar, Xdatcar

graph_params={
        'figure.figsize' : (6.432, 4.923),
        'font.family' : 'serif',
        'font.serif' : 'Times New Roman',
        "mathtext.fontset":'stix',
        'font.style':'normal',
        'font.weight':'bold',
        'font.size': 15,
        'axes.labelsize' : 25,
        'axes.labelweight' : 'bold',
        'axes.linewidth' : 3,
        'axes.facecolor' : 'none',
        'xtick.direction': 'in',
        'xtick.major.size' : 6,
        'xtick.major.width' : 2,
        'xtick.major.pad' : 5,
        'xtick.minor.visible' : False,
        'xtick.minor.size' : 4,
        'xtick.minor.width' : 2,
        'ytick.direction': 'in',
        'ytick.major.size' : 6,
        'ytick.major.width' : 2,
        'ytick.major.pad' : 5,
        'ytick.minor.visible' : False,
        'ytick.minor.size' : 4,
        'ytick.minor.width' : 2,
        'lines.linewidth': 2,
        'legend.frameon': False,
        'legend.facecolor': 'none',
        'savefig.bbox' : 'tight',
        'savefig.facecolor' : 'none'
        }
rcParams.update(graph_params)


class vaspresult:
    """
    Result of one vasp calculation
    """

    def __init__(self, result_dir: str="./"):

        self.result_dir = result_dir
        '''Dir of result files'''

        self.basename = os.path.basename(os.path.abspath(result_dir))
        '''Name of the calculation from the dir basename'''

        self.steps = vaspresult.stepdata(result_dir=result_dir)
        '''Data of ion steps during calculation'''

    def saveSteps(self):
        '''
        To plot the step parameters and save figs and save data into a csv file
        '''
        self.steps.saveSteps()

    class stepdata:
        '''
        Data of ion steps during calculation
        '''

        def __init__(self, result_dir: str="./"):

            self.result_dir = result_dir
            '''Dir of result files'''

            self.basename = os.path.basename(os.path.abspath(result_dir))

            self.step_energies = None
            '''Energy for each ion step'''

            self.step_forces = None
            '''Max force for each ion step'''

            self.step_distances = None
            '''Root mean square distance for each ion step from initial state'''

            self.step_volumes = None
            '''Volume of a cell for each ion step'''

            self._initStepData()

        def saveSteps(self):
            '''
            To plot the step parameters and save figs and save data into a csv file
            '''
            ion_steps = np.arange(1, self.step_energies.shape[0] + 1)
            plotData(ion_steps, self.step_energies, 
                    os.path.join(self.result_dir, f'step_energy_{self.basename}.svg'),
                    'Ion Step', 'Energy (eV)')
            plotData(ion_steps, self.step_forces, 
                    os.path.join(self.result_dir, f'step_force_{self.basename}.svg'),
                    'Ion Step', 'Max Force (eV/A)')
            plotData(ion_steps, self.step_distances, 
                    os.path.join(self.result_dir, f'step_distance_{self.basename}.svg'),
                    'Ion Step', 'RMSD From Initial (A)')
            plotData(ion_steps, self.step_volumes, 
                    os.path.join(self.result_dir, f'step_volume_{self.basename}.svg'),
                    'Ion Step', 'Cell Volume (A^3)')
            
            steps = np.array((ion_steps, self.step_energies, self.step_forces, self.step_distances, self.step_volumes))
            np.savetxt(os.path.join(self.result_dir, f'{self.basename}_step.csv'), steps.T, comments=' ',
                    delimiter=',', header='Ion Step,Energy,Max Force,RMS Distance,Cell Volume\n,eV,eV/A,A,A^3\n')

        def _initStepData(self):
            '''
            To load and process the step data
            '''
            self._getStepDistances()
            self._getStepEnergies()
            self._getStepForces()
            self._getStepVolumes()

        def _getStepEnergies(self):
            '''
            To get the energy for each ion step
            '''
            with open(os.path.join(self.result_dir, 'OSZICAR'), 'r') as f:
                lines = f.readlines()
            energies = []
            for line in lines:
                if "E0" in line:
                    parts = line.split()
                    energies.append(float(parts[4]))
            self.step_energies = np.array(energies)

        def _getStepForces(self):
            '''
            To get the max force for each ion step.
            '''
            with open(os.path.join(self.result_dir, 'OUTCAR'), 'r') as o:
                outdata = o.readlines()
            atoms = read(os.path.join(self.result_dir, 'CONTCAR'))

            fmax_list = []
            f = []
            for x, line in enumerate(outdata):
                if "TOTAL-FORCE" in line:
                    f = []
                    for y in range(x+2, len(outdata)):
                        if outdata[y][1] == "-":
                            break
                        f.append(outdata[y].split()[3:6])
                    try:
                        c = atoms._get_constraints()
                        indices_fixed = c[0].index    # the indices of atoms fixed
                        for i in indices_fixed:
                            f[i] = [0,0,0]
                    except:
                        pass
                    fmax = 0
                    for i in f:
                        fval = (float(i[0])**2 + float(i[1])**2 + float(i[2])**2)**(1./2.)
                        if fval > fmax:
                            fmax = fval
                    fmax_list.append(fmax)
            self.step_forces = np.array(fmax_list)

        def _getStepDistances(self):
            '''
            To get the root mean square distance for each ion step from initial state.
            '''
            initial_structure = Poscar.from_file(os.path.join(self.result_dir, 'POSCAR')).structure
            xdatcar = Xdatcar(os.path.join(self.result_dir, 'XDATCAR'))
            structures = xdatcar.structures
            distances = []
            for structure in structures:
                rmsd = np.sqrt(((structure.frac_coords - initial_structure.frac_coords) ** 2).sum(axis=1).mean())
                distances.append(rmsd)
            self.step_distances = np.array(distances)

        def _getStepVolumes(self):
            '''
            To get the cell volume for each ion step.
            '''
            with open(os.path.join(self.result_dir, 'OUTCAR'), 'r') as f:
                lines = f.readlines()
            volumes = []
            for line in lines:
                if "volume of cell" in line:
                    parts = line.split()
                    volumes.append(float(parts[-1]))
            self.step_volumes = np.array(volumes[1:])   # From ion step 1

def plotData(x, y, fig_dir, xlab, ylab):
    '''
    To plot data and save an svg fig
    '''
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(x, y, '-o')
    plt.savefig(fig_dir)
    plt.cla()


if __name__ == '__main__':
    vasp_data = vaspresult()
    vasp_data.saveSteps()
