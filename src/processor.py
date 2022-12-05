import utm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import xgboost as xg 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.linear_model import Ridge

sns.set_style("whitegrid")

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=BIGGER_SIZE)
plt.rc('ytick', labelsize=BIGGER_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE) 

class PNEUMATrajectory:
    def __init__(self, PATH):
        self.path = PATH
    
    def load_data(self):
        '''To read the csv file
        '''
        self.df = pd.read_csv(self.path)

    def save_data(self, output_path):
        '''To read the csv file
        '''
        self.df.to_csv(output_path, index=None)

    def filter_data(self, data, v_id):
        '''To filter the dataframe for a specific vehicle
        '''
        self.df = data[data.id==v_id].copy()
        self.df.reset_index(inplace=True)

    def getter(self, attribute):
        '''To get the data for a specific attribute
        '''
        return list(self.df[attribute])

    def setter(self, attribute, list_value):
        '''To set the value for a specific attribute
        '''
        self.df[attribute] = list_value

    def get_derivative(self, attribute):
        '''To obtain the derivative at a fixed spacing i.e., 
        frame interval
        '''
        return np.round(np.gradient(self.df[attribute], 0.04),4)

    def get_azimuth(self):
        # do something
        raise NotImplementedError
    
    def plot_attribute(self, 
                        attribute, 
                        label, 
                        color, 
                        ax, 
                        alpha=1):
        '''To plot an attribute
        '''
        ax.plot(self.df['frame'], 
                self.df[attribute], 
                c=color, 
                label = label,
                alpha=alpha)
        return ax

    def check_sign_change(self, attribute, threshold=25):
        '''To check the number of sign changes with a
        threshold time (in frames) such as 25 frames of 1 second.
        If the difference betweeen the two sign changes is less then
        25 frames or 1 s, it returns True
        '''
        sign_toggle = np.where(np.diff(np.sign(self.df[attribute])))[0]
        if np.min(np.diff(sign_toggle))< threshold:
            return True
        else:
            return False

    def smoother(self, 
                attribute, 
                window,
                algorithm = 'gaussian', 
                plot=False,
                *kwargs
                ):	
        '''Using off-shelf smoothing on the specified
        attribute series
        '''
        if algorithm == "gaussian":
            sm_attribute = gaussian_filter1d(self.df[attribute], window)

        elif algorithm == "sma":
            sm_attribute = pd.Series(self.df[attribute]).rolling(min_periods=1, 
                                                                center=True, 
                                                                window=window).mean()
        elif algorithm == "sg":
            # same as moving average for polyorder 1
            sm_attribute = savgol_filter(list(self.df[attribute]),window_length= window, 
                                    polyorder=1, mode='nearest')
        else:
            raise("Please enter a valid algorithm")
        return sm_attribute
    

    def detect_anomalies(self, 
                        n = 2, 
                        b = 15,
                        model='xgb'):
        '''Using a ML model to reconstruct the series of interest
        Note that the regularization for the boosting model is dependent on 
        max value of the acceleration
        '''
        X, y = self.df[['sv', 'a_y', 'dsv']], self.df['dsv']    
        train_X, _, train_y, _ = train_test_split(X, y, 
                            test_size = 0.2, random_state = 123) 
        l2 = b*np.power(np.abs(np.max(self.df.dsv)), n)
        if model=="xgb":
            xgb_r = xg.XGBRegressor(objective ='reg:squarederror',
                                    n_estimators  = 300, 
                                    seed = 3, 
                                    reg_lambda=l2, 
                                    njobs=2)   
            xgb_r.fit(train_X, train_y) 
            return xgb_r.predict(X)
        elif model=="linear":
            regr = Ridge(alpha=l2)
            regr.fit(train_X, train_y)
            return regr.predict(X)

    def plot_process(self, image_path, anomaly_index):
        '''to plot the speeds and accelerations before and after treatment
        '''
        fig, ax = plt.subplots(2, 2, figsize=(14,6))
        ax[0,0].plot(self.df.frame, self.df.dv, label="Original", linestyle= "-", color='b', alpha=0.6)
        ax[0,0].plot(self.df.frame, self.df.dsv, label="SGF with anomalies", color='k')
        ax[0,1].plot(self.df.frame, self.df.a_x, label="Original", linestyle= "-", alpha=0.6)
        ax[0,1].plot(self.df[np.abs(self.df.xgb-self.df.dsv)>tol].frame, 
                     self.df[np.abs(self.df.xgb-self.df.dsv)>tol].dsv, 'rx', label="Anomalies")
        ax[0,1].plot(self.df.frame, self.df.xgb, label= 'Anomaly Mask', color='k')

        if len(anomaly_index)!=0:
            ax[1,0].plot(self.df.frame, self.df.v, label='Original', alpha=0.6)
            ax[1,0].plot(self.df.frame, self.df.g_s, label='GF with anomalies', color='r')
            ax[1,0].plot(self.df.frame, self.df.g_ns, label='GF without anomalies', color='k')

            ax[1,1].plot(self.df.frame, self.df.dv, label='Original', alpha=0.6)
            ax[1,1].plot(self.df.frame, self.df.gdv, label='GF with anomalies', color='r')
            ax[1,1].plot(self.df.frame, self.df.g_na, label='GF without anomalies', color='k')

            xmin,xmax = np.min(self.df[np.abs(self.df.xgb-self.df.dsv)>tol].frame-50),\
            np.max(self.df[np.abs(self.df.xgb-self.df.dsv)>tol].frame+50)
            ax[0,0].set_xlim([xmin, xmax])
            ax[0,1].set_xlim([xmin, xmax])
            ax[1,0].set_xlim([xmin, xmax])
            ax[1,1].set_xlim([xmin, xmax])   

        else:
            ax[1,0].plot(self.df.frame, self.df.v, label='Original', alpha=0.6)
            ax[1,0].plot(self.df.frame, self.df.sv, label="SGF", color='k')

            ax[1,1].plot(self.df.frame, self.df.dv, label='Original', alpha=0.6)
            ax[1,1].plot(self.df.frame, self.df.dsv, label="SGF", color='k')

        ax[0,0].set_ylabel("Acceleration ($m/s^2$)")
        ax[0,1].set_ylabel("Acceleration ($m/s^2$)")
        ax[1,0].set_ylabel("Speed ($Km/h$)")
        ax[1,1].set_ylabel("Acceleration ($m/s^2$)")
        ax[1,0].set_xlabel("Time (s)")
        ax[1,1].set_xlabel("Time (s)")

        ax[0,0].grid(True)
        ax[0,0].set_xticklabels([])
        ax[0,1].set_xticklabels([])

        ax[0,0].yaxis.set_tick_params(labelsize=15)
        ax[0,1].yaxis.set_tick_params(labelsize=15)
        ax[1,0].yaxis.set_tick_params(labelsize=15)
        ax[1,1].yaxis.set_tick_params(labelsize=15)

        ax[0,0].xaxis.set_tick_params(labelsize=15)
        ax[0,1].xaxis.set_tick_params(labelsize=15)
        ax[1,0].xaxis.set_tick_params(labelsize=15)
        ax[1,1].xaxis.set_tick_params(labelsize=15)

        ax[0,0].legend(loc="upper right", bbox_to_anchor=(1,1.05),handletextpad=0.1,prop={'size': 14},labelspacing = .2)
        ax[0,1].legend(loc="upper left", bbox_to_anchor=(0,1.05),handletextpad=0.1,prop={'size': 14},labelspacing = .2)
        ax[1,0].legend(loc="upper left", bbox_to_anchor=(0,1.05),handletextpad=0.1,prop={'size': 14},labelspacing = .2)	
        ax[1,1].legend(loc="upper left", bbox_to_anchor=(0,1.05),handletextpad=0.1,prop={'size': 14},labelspacing = .2)
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax[0,0].text(0.35, 1.05, "1. Denoise", transform=ax[0,0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax[0,1].text(0.35, 1.05, "2. Remove anomalies", transform=ax[0,1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax[1,0].text(0.35, 1.05, "3. Retrieve consistent speeds", transform=ax[1,0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props, zorder=10)
        ax[1,1].text(0.35, 1.05, "4. Re-calculate accelerations", transform=ax[1,1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props, zorder=10)
        xmin,xmax = np.min(self.df[np.abs(self.df.xgb-self.df.dsv)>tol].frame-50),\
                  np.max(self.df[np.abs(self.df.xgb-self.df.dsv)>tol].frame+50)
        # ax[0,0].set_xlim([xmin, xmax])
        # ax[0,1].set_xlim([xmin, xmax])
        # ax[1,0].set_xlim([xmin, xmax])
        # ax[1,1].set_xlim([xmin, xmax])
        plt.suptitle("Vehicle id: %d"%int(self.df.id.unique()[0]))
        plt.tight_layout()
        plt.savefig(image_path+str(self.df.id.unique()[0])+".png", dpi=400)
        fig.clear()
        plt.close(fig)

if __name__ == "__main__":
    
    INPUT_DATA = 	"../data/sample_data.csv"
    OUTPUT_PLOTS = 	"../plots/"
    OUTPUT_DATA = 	"../data/derived/"
    DERIVED_DATA = 	"../data/derived/"
    
    ## define parameters of the processing
    smoothing_window = 25 			# smoothing window 
    smoothing_window_gaussian = 12
    tol = .1 						# to check the reconstruction loss in the acceleration subsequence
    merge_subsequences = 10 		# to merge subsequences if they are within these many frames apart

    
    data = pd.read_csv(INPUT_DATA)
    # data['frame'] = data['frame'].str.rstrip(';').astype(float)
    all_ids = data.id.unique()
    
    all_ids = sorted(all_ids, reverse=True)
    print(len(all_ids))

    for v_id in tqdm(list(all_ids)):
        vehicle = PNEUMATrajectory("dummy")
        vehicle.filter_data(data, v_id)
        try:
            vehicle.setter('dv', vehicle.get_derivative('v')/3.6)
        except Exception as e:
            print(e)
            pass
            # raise(e)
        
        # apply smoothing filter to the speed
        vehicle.setter('sv', vehicle.smoother('v', window=smoothing_window, algorithm='sg'))

        # convert Positions from LAT-LON to UTM
        vehicle.setter('northing', vehicle.df.apply(lambda x: np.round(utm.from_latlon(x['lat'], x['lon'])[1],8), axis=1))
        vehicle.setter('easting', vehicle.df.apply(lambda x: np.round(utm.from_latlon(x['lat'], x['lon'])[0],8), axis=1))

        # calculate the speed in Km/h
        try:
            vehicle.df['pos_v'] = np.round(3.6*np.sqrt(vehicle.get_derivative('northing')**2 + vehicle.get_derivative('easting')**2),4)
            vehicle.setter('dsv', vehicle.get_derivative('sv')/3.6)
        except Exception as e:
            # when the array is too small for a derivative
            print(e)
            pass
            # raise(e)

        try:
            # get the reconstructed series
            vehicle.setter('xgb', vehicle.detect_anomalies()) #default params: b=4, n=3

            # check reconstruction loss with respect to the tolerance param
            vehicle.setter('filtered_acc', vehicle.df.apply(lambda x: x['dsv'] if np.abs(x.dsv-x.xgb)<tol else x.xgb, axis=1))

            # Calculate speeds consistent with the new accelerations
            t_acc = np.array(vehicle.getter('filtered_acc')) 
            t_speed = list(np.array(vehicle.getter('sv'))/3.6)
            # ns = t_speed[0] + 2 * 0.04* np.c_[np.r_[0, t_acc[1:-1:2].cumsum()], 
            # 								t_acc[::2].cumsum() - t_acc[0] / 2].ravel()[:len(t_acc)]
            ns = t_speed[0] +  0.04 * (t_acc + np.r_[0, t_acc[:-1]]).cumsum()/2
            vehicle.setter('filtered_speed', ns*3.6)

            # assign anomalies based on the reconstructed error	
            vehicle.setter('anomaly', vehicle.df.apply(lambda x: 0 if np.abs(x.dsv-x.xgb)<tol else 1, axis=1))

            ano_index = list(vehicle.df[vehicle.df.anomaly==1].index)
            new_indices = []
            # merge subsequences where anomalies are close to each other e.g., within 10 frames
            for k, j in enumerate(ano_index):
                if k==0:
                    continue
                if ano_index[k]-ano_index[k-1] < merge_subsequences:
                    new_indices.extend(list(range(ano_index[k-1], ano_index[k])))
            for j in new_indices:
                vehicle.df.loc[j, 'anomaly']=1

            if len(ano_index)!=0:
                # if anomalies are detected, do this:

                # replace old speeds during anomalous subsequences with the new speeds
                vehicle.setter('new_speed', vehicle.df.apply(lambda x: x['v'] if x.anomaly==0 else x.filtered_speed, axis=1))

                # apply gaussian filter on the old accelerations
                vehicle.setter('gdv', vehicle.smoother('dv', smoothing_window_gaussian))

                # apply SG filter on the new speeds
                vehicle.setter('sg_ns',vehicle.smoother('new_speed', smoothing_window, algorithm="sg"))

                # apply Gaussian filter on the old speeds
                vehicle.setter('g_s', vehicle.smoother('v', smoothing_window_gaussian))

                # apply Gaussian filter on the new speeds
                vehicle.setter('g_ns', vehicle.smoother('new_speed', smoothing_window_gaussian))

                # Calculate new accelerations from smoothed new speeds
                vehicle.setter('sg_na', vehicle.get_derivative('sg_ns')/3.6)
                vehicle.setter('g_na', vehicle.get_derivative('g_ns')/3.6)

            else:
                # else when anomalies are not detected, do this:

                # replace old speeds during anomalous subsequences with the new speeds
                vehicle.setter('new_speed', vehicle.df.apply(lambda x: x['sv'], axis=1))

                vehicle.setter('gdv', vehicle.df.apply(lambda x: x['dsv'], axis=1))

                vehicle.setter('sg_ns',vehicle.df.apply(lambda x: x['sv'], axis=1))

                vehicle.setter('g_s', vehicle.df.apply(lambda x: x['sv'], axis=1))

                vehicle.setter('g_ns', vehicle.df.apply(lambda x: x['sv'], axis=1))

                vehicle.setter('sg_na', vehicle.df.apply(lambda x: x['dsv'], axis=1))
                vehicle.setter('g_na', vehicle.df.apply(lambda x: x['dsv'], axis=1))

            temp_save = pd.DataFrame(vehicle.df[["id", "frame", "g_na", "g_ns"]])
            temp_save.columns = ["id", "frame", "g_na", "g_ns"]
            temp_save.to_csv(OUTPUT_DATA+str(v_id)+".csv")
            
            vehicle.plot_process(OUTPUT_PLOTS, ano_index)
            vehicle.save_data(OUTPUT_DATA + str(v_id)+".csv")

        except ValueError as e:
            print(e)
            print("Vehicle not present %d" %v_id)
            # pass
            raise(e)
        except Exception as e:
            print(e)
            pass
            # raise(e)
