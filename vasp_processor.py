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

    def __init__(self, result_dir: str='./'):

        self.result_dir = result_dir
        '''Directory of result files'''

        self.basename = os.path.basename(os.path.abspath(result_dir))
        '''Name of the calculation from the dir basename'''

        self.steps = vaspresult.stepdata(result_dir=result_dir)
        '''Data of ion steps during calculation'''

    def saveSteps(self, dir=None):
        '''
        To plot the step parameters and save figs and save data into a csv file
        '''
        self.steps.plotSteps(dir)
        self.steps.saveSteps(dir)

    class stepdata:
        '''
        Data of ion steps during calculation
        '''

        data_type = ['step_energies', 'step_forces', 'step_distances', 'step_volumes']

        data_title ={ 'step_energies': 'Energy',
                      'step_forces': 'Max Force',
                      'step_distances': 'RSM Distance',
                      'step_volumes': 'Cell Volume' }
        
        data_unit = { 'step_energies': 'eV',
                      'step_forces': 'eV/A',
                      'step_distances': 'A',
                      'step_volumes': 'A^3' }

        def __init__(self, result_dir: str="./"):

            self.result_dir = result_dir
            '''Dir of result files'''

            self.basename = os.path.basename(os.path.abspath(result_dir))
            '''Name of the calculation from the dir basename'''

            self.ion_steps = None
            '''Ion steps as an array, from 1 to the last step'''
            
            self.step_energies = None
            '''Energy for each ion step'''

            self.step_forces = None
            '''Max force for each ion step'''

            self.step_distances = None
            '''Root mean square distance for each ion step from initial state'''

            self.step_volumes = None
            '''Volume of a cell for each ion step'''
            
            self._initStepData()

        def plotSteps(self, dir: str=None):
            '''
            To plot the parameters during ion steps save figs
            '''
            if dir is None:
                dir = self.result_dir

            for data in vasp_data.stepdata.data_type:
                exec(f'singlePlot(self.ion_steps, self.{data}, \
                        os.path.join(dir, "{data}_{self.basename}.svg"), \
                        "Ion Step", \
                        "{vasp_data.stepdata.data_title[data]} ({vasp_data.stepdata.data_unit[data]})")')

        def saveSteps(self, dir: str=None):
            '''
            To save the parameters during ion steps into a csv file
            '''
            if dir is None:
                dir = self.result_dir
            steps = np.array((self.ion_steps, self.step_energies, self.step_forces, self.step_distances, self.step_volumes))
            np.savetxt(os.path.join(dir, f'{self.basename}_step.csv'), steps.T, comments=' ', delimiter=',', 
                    header=f'Ion Step,{",".join(vasp_data.stepdata.data_title.values())}\n,{",".join(vasp_data.stepdata.data_unit.values())}\n{f",{self.basename}" * 4}\n')

        def _initStepData(self):
            '''
            To load and process the step data
            '''
            self._getStepDistances()
            self._getStepEnergies()
            self._getStepForces()
            self._getStepVolumes()
            self.ion_steps = np.arange(1, self.step_energies.shape[0] + 1)

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


class vaspgroup:
    '''
    Results of a group of vasp calculations
    '''

    def __init__(self, group_dir: str='./'):
        
        self.group_dir = group_dir
        '''Directory of the group'''

        self.group_name = os.path.basename(group_dir)
        
        self.result_dirs = [f.path for f in os.scandir(group_dir) if f.is_dir()]
        '''Directory of the calculation results in the group'''

        self.results = [vasp_data(d) for d in self.result_dirs]
        '''Calculation results as a list'''

    def stepCompare(self):
        '''
        To compare the ion steps among different calculaitons
        '''
        pass


        


def singlePlot(x: np.array, y: np.array, fig_dir: str, xlab: str, ylab: str):
    '''
    To plot data and save an svg fig
    '''
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(x, y, '-o')
    plt.savefig(fig_dir)
    plt.cla()


def multiPlot(xys: list, fig_dir: str, xlab: str, ylab: str):
    '''
    To plot multiple data in an svg fig
    '''
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    for xy in xys:
        plt.plot(xy[0], xy[1], '-o')
    plt.savefig(fig_dir)
    plt.cla


if __name__ == '__main__':
    vasp_data = vaspresult()
    vasp_data.saveSteps()
