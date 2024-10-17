import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

#This is a program that finds the average displacement across Socompa at each aquasition and then fits an equation with two different gradients with a change between them at t_0 from data proccessed by Lui et al 2023

#Read in files
descendingFrame = mat73.loadmat("D:\PhD\InSAR_156D\InSAR_156D\Data_Desc.mat")
ascendingFrame = mat73.loadmat("D:\PhD\InSAR_149A\InSAR_149A\Data_Asc.mat")

#Data from https://zenodo.org/records/7688945

def get_average_deformation(frame, lat0=24.35, lat1=24.45, lon0=68.20, lon1=68.30):
    """
    Averages the deformation at each aquation over Socompa (or within any area defined by 4 points in the SE quadsphere)
    
    Inputs:
    frame:  the proccessed deformation time-serires. A dictionary with keys "lat", "lon" (the latitude and longatude of the pixel), "day" (days after first epoc aquasition happened) and "ifg_aps" (deformation timeseries GACOS collected)
    lat0, lat1: the minimum and maximum latitude over the intrest region. 0.05 decimal degrees eitherside of Socompa peak by defalut
    lon0, lon1: the minimum and maximum longitude over the intrest region. 0.05 decimal degrees eitherside of Socompa peak by defalut
    
    Outputs:
    averagedFrame: dictionary with keys "ifg_aps" (the GACOS corrected timeseries of averaged deformation over Socompa) and "day" (the number of days after the first epoc)
    """
    
    #Finding indecies that are within the frame over Socompa defined by input
    lat=np.bitwise_and(abs(frame["lat"])>lat0,abs(frame["lat"])<lat1)#list element are 1 for pixels in lat range
    lon=np.bitwise_and(abs(frame["lon"])>lon0,abs(frame["lon"])<lon1)#list element are 1 for pixels in lon range
    socompaLoc=np.bitwise_and(lat,lon)
    indexesInRange=np.where(socompaLoc==1)#find indexs where pixel is within 4 points
    
    #Average the deformation in the track over Socompa
    averageDef=np.average(frame["ifg_aps"][indexesInRange[0]],0)
    
    #Set up output dictionary averagedFrame
    averagedFrame={}
    averagedFrame["ifg_aps"]=[averageDef]
    averagedFrame["day"]=frame["day"]

    return averagedFrame

def heaviside(x,x0):
    """
    The Heaviside function
    
    Inputs:
    x:The x coordinate of the data
    x0:The x coordiate of the step from 0 to 1
    
    Outputs:
    y: =0 for x<x0 and =1 for x>=x0
    """
    x=x-x0
    return np.heaviside(x,1)

def deformation_from_0(t,t0,v2,c1):
    """
    A fitting function for a timeseries with no deformation until t0 then a linear deformation rate of v2 after
    
    Inputs:
    t: the time after first epoc
    t0: the onset of deformation
    v2: the deformation rate after t0
    c1: the intercept
    
    Outputs:
    y: =c1 for t<t0 and =v2(t-t0) +c1 for f>=t0
    """
    c2 = t0*(-v2)
    
    before_t0 = c1
    after_t0 = v2*t + c2
    return before_t0 + heaviside(t,t0)*after_t0

def deformation_from_v1(t,t0,v2,c1,v1):
    """
    A fitting function for a timeseries with a deformation rate of v1 until t0 then a linear deformation rate of v1+v2 after
    
    Inputs:
    t: the time after first epoc
    t0: the onset of deformation
    v2: the deformation rate after t0 (Note v2 + v1 is total deformation rate after t0)
    c1: the intercept
    v1: the initial deformation rate before t0
    
    Outputs:
    y: =v1+c1 for t<t0 and =v2(t-t0) + (v1*t+c1) for f>=t0
    """
    c2 = t0*(-v2)
    
    before_t0 = v1*t + c1
    after_t0 = v2*t + c2
    return before_t0 + heaviside(t,t0)*after_t0
frames={"ascending":get_average_deformation(ascendingFrame),"descending":get_average_deformation(descendingFrame)} #the ascending and descending frame
fits={"ascending":[], "descending":[]}#keeps the results of fitting. Stores output of scipy.optimize.curve_fit() in fits[track][initalDef] where track is the string "ascending" or "descending" and initalDef is 0 for fitting with no initial deformation and 1 is fitting that considers a non-0 deformation rate defore onset time

fits["ascending"].append(curve_fit(deformation_from_0,frames["ascending"]["day"],frames["ascending"]["ifg_aps"][0],[800,-1/250,1]))
fits["ascending"].append(curve_fit(deformation_from_v1,frames["ascending"]["day"],frames["ascending"]["ifg_aps"][0],[800,-1/250,1,0]))

fits["descending"].append(curve_fit(deformation_from_0,frames["descending"]["day"],frames["descending"]["ifg_aps"][0],[800,-1/250,1]))
fits["descending"].append(curve_fit(deformation_from_v1,frames["descending"]["day"],frames["descending"]["ifg_aps"][0],[800,-1/250,1,0]))

errors={"ascending":[], "descending":[]}#errors of fitting calculated from covarience will be stored in here

errors["ascending"].append(np.sqrt(np.diag(fits["ascending"][0][1])))
errors["ascending"].append(np.sqrt(np.diag(fits["ascending"][1][1])))

errors["descending"].append(np.sqrt(np.diag(fits["descending"][0][1])))
errors["descending"].append(np.sqrt(np.diag(fits["descending"][1][1])))

residuals={"ascending":[],"descending":[]}#residuals from the fittings

residuals["ascending"].append(frames["ascending"]["ifg_aps"][0]-deformation_from_0(frames["ascending"]["day"],*fits["ascending"][0][0]))
residuals["ascending"].append(frames["ascending"]["ifg_aps"][0]-deformation_from_v1(frames["ascending"]["day"],*fits["ascending"][1][0]))

residuals["descending"].append(frames["descending"]["ifg_aps"][0]-deformation_from_0(frames["descending"]["day"],*fits["descending"][0][0]))
residuals["descending"].append(frames["descending"]["ifg_aps"][0]-deformation_from_v1(frames["descending"]["day"],*fits["descending"][1][0]))

#A linspace to plot the fit
timeAxis=np.linspace(np.min(frames["ascending"]["day"]),np.max(frames["ascending"]["day"]),1000)

#Create figure for fits and resudals
fig, ax = plt.subplots(4,2,figsize=(16, 16))

#setting up parameters ready for the loop
trackDirec=("ascending","descending")#the ascending and descending track
labels=("Fit with no initial displacement","Fit with initial displacement")#The label for the graphs of the fits

minOf={"ascending":min(frames["ascending"]["ifg_aps"][0]),"descending":min(frames["descending"]["ifg_aps"][0])}#The minimum of the timeseries to draw the verticle lines
maxOf={"ascending":max(frames["ascending"]["ifg_aps"][0]),"descending":max(frames["descending"]["ifg_aps"][0])}

onsetTimes={"ascending":[882,685],"descending":[875,678]}#The days of the EQ and the onset predicted by Lui et al., for the veritlae lines below

for i in range(len(trackDirec)):
    direc=trackDirec[i]#Current direction
    for fitToInitial in (0,1):
        
        #Plot Deformation
        
        axIndex = i*2+fitToInitial#Which index is this graph in ax
        fitLabel = labels[fitToInitial]#The label for the fit
        
        ax[axIndex][0].set_title("Average Deformation over Socompa for Ascending Track")
        ax[axIndex][0].set_xlabel("Time since First Epoc (days)")
        ax[axIndex][0].set_ylabel("LOS Displacement (mm)?")
        ax[axIndex][0].plot(frames[direc]["day"],frames[direc]["ifg_aps"][0],".")
        if not fitToInitial:#Does this fit have no initial deforation or (else) does it have an initial deformation
            ax[axIndex][0].plot(timeAxis,deformation_from_0(timeAxis,*fits[direc][fitToInitial][0]),label=fitLabel)
        else:
            ax[axIndex][0].plot(timeAxis,deformation_from_v1(timeAxis,*fits[direc][fitToInitial][0]),label=fitLabel)
        ax[axIndex][0].vlines(onsetTimes[direc][0],minOf[direc],maxOf[direc],"c","dashed", label="M6.8 Earthquake")
        ax[axIndex][0].vlines(onsetTimes[direc][1],minOf[direc],maxOf[direc],"g","dashed", label="Liu et al. onset time")
        ax[axIndex][0].legend()       
        
        #Plot Residuals
        
        x=frames[direc]["day"]
        y=residuals[direc][fitToInitial]
        ax[axIndex][1].set_title("Residual")
        ax[axIndex][1].set_xlabel("Time since First Epoc (days)")
        ax[axIndex][1].set_ylabel("Residual (mm)?")
        ax[axIndex][1].plot(x,y, "dimgray")
        ax[axIndex][1].fill_between(x, y, where=(y > 0), color='blue', alpha=0.5)
        ax[axIndex][1].fill_between(x, y, where=(y < 0), color='red', alpha=0.5)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#Original code to plot deformation without residuals

print("Ascending")

plt.rcParams['figure.figsize'] = [9,7]

plt.title("Average Deformation over Socompa for Ascending Track")
plt.xlabel("Time since First Epoc (days)")
plt.ylabel("LOS Displacement (mm)?")
plt.plot(frames["ascending"]["day"],frames["ascending"]["ifg_aps"][0],".")
plt.plot(timeAxis,deformation_from_0(timeAxis,*fits["ascending"][0][0]),label="Fit with no initial displacement")
plt.vlines(882,minOf["ascending"],maxOf["ascending"],"c","dashed", label="M6.8 Earthquake")
plt.vlines(685,minOf["ascending"],maxOf["ascending"],"g","dashed", label="Liu et al. onset time")
plt.legend()
plt.show()

print(f"Onset time: {fits['ascending'][0][0][0]:.2f} \u00B1 {errors['ascending'][0][0]:.2f} Days")
print(f"Post onset deformation: {fits['ascending'][0][0][1]*365:.2f} \u00B1 {errors['ascending'][0][1]*365:.2f} mm/Year ?")

plt.title("Average Deformation over Socompa for Ascending Track")
plt.xlabel("Time since First Epoc (days)")
plt.ylabel("LOS Displacement (mm)?")
plt.plot(frames["ascending"]["day"],frames["ascending"]["ifg_aps"][0],".")
plt.plot(timeAxis,deformation_from_v1(timeAxis,*fits["ascending"][1][0]),label="Fit with initial displacement")
plt.vlines(882,minOf["ascending"],maxOf["ascending"],"c","dashed", label="M6.8 Earthquake")
plt.vlines(685,minOf["ascending"],maxOf["ascending"],"g","dashed", label="Liu et al. onset time")
plt.legend()
plt.show()

print(f"Onset time: {fits['ascending'][1][0][0]:.2f} \u00B1 {errors['ascending'][1][0]:.2f} Days")
print(f"Post onset deformation: {fits['ascending'][1][0][1]*365:.2f} \u00B1 {errors['ascending'][1][1]*365:.2f} mm/Year ?")

print("Descending")

plt.title("Average Deformation over Socompa for Descending Track")
plt.xlabel("Time since First Epoc (days)")
plt.ylabel("LOS Displacement (mm)?")
plt.plot(frames["descending"]["day"],frames["descending"]["ifg_aps"][0],".")
plt.plot(timeAxis,deformation_from_0(timeAxis,*fits["descending"][0][0]),label="Fit with no initial displacement")
plt.vlines(875,minOf["descending"],maxOf["descending"],"c","dashed", label="M6.8 Earthquake")
plt.vlines(678,minOf["descending"],maxOf["descending"],"g","dashed", label="Liu et al. onset time")
plt.legend()
plt.show()

print(f"Onset time: {fits['descending'][0][0][0]:.2f} \u00B1 {errors['descending'][0][0]:.2f} Days")
print(f"Post onset deformation: {fits['descending'][0][0][1]*365:.2f} \u00B1 {errors['descending'][0][1]*365:.2f} mm/Year ?")

plt.title("Average Deformation over Socompa for Descending Track")
plt.xlabel("Time since First Epoc (days)")
plt.ylabel("LOS Displacement (mm)?")
plt.plot(frames["descending"]["day"],frames["descending"]["ifg_aps"][0],".")
plt.plot(timeAxis,deformation_from_v1(timeAxis,*fits["descending"][1][0]),label="Fit with initial displacement")
plt.vlines(875,minOf["descending"],maxOf["descending"],"c","dashed", label="M6.8 Earthquake")
plt.vlines(678,minOf["descending"],maxOf["descending"],"g","dashed", label="Liu et al. onset time")
plt.legend()
plt.show()

print(f"Onset time: {fits['descending'][1][0][0]:.2f} \u00B1 {errors['descending'][1][0]:.2f} Days")
print(f"Post onset deformation: {fits['descending'][1][0][1]*365:.2f} \u00B1 {errors['descending'][1][1]*365:.2f} mm/Year ?")
