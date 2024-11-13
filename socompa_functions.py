# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:56:23 2024

@author: py20bk
"""

import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import folium
from IPython.display import display, IFrame
import copy
from datetime import datetime
import pygmt


#This is a program that finds the average displacement across Socompa at each aquasition and then fits an equation with two different gradients with a change between them at t_0 from data proccessed by Lui et al 2023

#Read in files
def get_socompa_data():
    #Data from https://zenodo.org/records/7688945
    descendingFrame = mat73.loadmat(r"C:\Users\py20bk\Downloads\InSAR_156D\InSAR_156D\Data_Desc.mat")
    ascendingFrame = mat73.loadmat(r"C:\Users\py20bk\Downloads\InSAR_149A\InSAR_149A\Data_Asc.mat")
    
    return ascendingFrame,descendingFrame

def get_socompa_location():
    return (-24.3959, -68.245997)

def get_onset_times():
    return (882,685)#The days of the EQ and the onset predicted by Lui et al. 2023

def get_first_epoc():
    return datetime(2018, 1, 3)

def within_rectangle(frame, latSouth, latNorth, lonEast, lonWest):
    #Finding indecies that are within the frame over Socompa defined by input
    
    lat=np.bitwise_and(frame["lat"]>latSouth,frame["lat"]<latNorth)#list element are 1 for pixels in lat range
    lon=np.bitwise_and(frame["lon"]>lonWest,frame["lon"]<lonEast)#list element are 1 for pixels in lon range
    socompaLoc=np.bitwise_and(lat,lon)

    #print(len(frame["lon"]))
    #print(len(lat))
    #print(len(lon))

    indexesInRange=np.where(socompaLoc==1)
    returnFrame = copy.deepcopy(frame)
    returnFrame["ifg_aps"]=frame["ifg_aps"][indexesInRange[0]]
    returnFrame["lon"]=frame["lon"][indexesInRange[0]]
    returnFrame["lat"]=frame["lat"][indexesInRange[0]]
    #frame[""]
    #print(frame.keys())

    #print(frame["lat"])
    #print(frame["lon"])
    #print(centre[0])

    return returnFrame

def get_average_deformation(frame, rad, centre,  ascending=True):
    """
    Averages the deformation at each aquation over Socompa (or within any area defined by 4 points in the SE quadsphere)
    
    Inputs:
    frame:  the proccessed deformation time-serires. A dictionary with keys "lat", "lon" (the latitude and longatude of the pixel), "day" (days after first epoc aquasition happened) and "ifg_aps" (deformation timeseries GACOS collected)
    lat0, lat1: the minimum and maximum latitude over the intrest region. 0.05 decimal degrees eitherside of Socompa peak by defalut
    lon0, lon1: the minimum and maximum longitude over the intrest region. 0.05 decimal degrees eitherside of Socompa peak by defalut
    
    Outputs:
    averagedFrame: dictionary with keys "ifg_aps" (the GACOS corrected timeseries of averaged deformation over Socompa) and "day" (the number of days after the first epoc)
    """
    
    distance=haveresine(frame["lat"],frame["lon"],centre[0],centre[1])
    socompaLoc = distance<=rad
    
    indexesInRange=np.where(socompaLoc==1)#find indexs where pixel is within 4 points

    numOfPoints=len(indexesInRange[0])
    
    #Average the deformation in the track over Socompa
    averageDef=np.average(frame["ifg_aps"][indexesInRange[0]],0)
    #Set up output dictionary averagedFrame
    averagedFrame={}
    averagedFrame["ifg_aps"]=[averageDef]
    if ascending:
        averagedFrame["day"]=frame["day"]
    else:
        averagedFrame["day"]=frame["day"]+7
    
    return averagedFrame, numOfPoints

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

def getFitParams(ascendingFrame, descendingFrame, plot=False):#Deleted Socompa Location

    """
    ### BEGINING OF CODE FROM COPILOT
        
    # Define the center location (Socompa Peak)
        
    # Convert side length to degrees (approximation)
    # 1 degree of latitude is approximately 111 km
    half_side_length_deg = (boxSize / 2) / 111
        
    # Calculate the coordinates of the corners
    lat, lon = socompaLocation
    north = lat + half_side_length_deg
    south = lat - half_side_length_deg
    east = lon + half_side_length_deg / np.cos(np.radians(lat))
    west = lon - half_side_length_deg / np.cos(np.radians(lat))
    bounds = [(south, west), (north, east)]
        
    #### END OF CODE FROM COPILOT
    """

    frames={"ascending":ascendingFrame, "descending":descendingFrame}
    
    fits={"ascending":[], "descending":[]}#keeps the results of fitting. Stores output of scipy.optimize.curve_fit() in fits[track][initalDef] where track is the string "ascending" or "descending" and initalDef is 0 for fitting with no initial deformation and 1 is fitting that considers a non-0 deformation rate defore onset time
    
    fits["ascending"].append(curve_fit(deformation_from_0,frames["ascending"]["day"],frames["ascending"]["ifg_aps"],[800,-1/250,1]))#rem[0]
    fits["ascending"].append(curve_fit(deformation_from_v1,frames["ascending"]["day"],frames["ascending"]["ifg_aps"],[800,-1/250,1,0]))
    
    fits["descending"].append(curve_fit(deformation_from_0,frames["descending"]["day"],frames["descending"]["ifg_aps"],[800,-1/250,1]))#rem[0]
    fits["descending"].append(curve_fit(deformation_from_v1,frames["descending"]["day"],frames["descending"]["ifg_aps"],[800,-1/250,1,0]))
    
    errors={"ascending":[], "descending":[]}#errors of fitting calculated from covarience will be stored in here
    
    errors["ascending"].append(np.sqrt(np.diag(fits["ascending"][0][1])))
    errors["ascending"].append(np.sqrt(np.diag(fits["ascending"][1][1])))
    
    errors["descending"].append(np.sqrt(np.diag(fits["descending"][0][1])))
    errors["descending"].append(np.sqrt(np.diag(fits["descending"][1][1])))
    
    residuals={"ascending":[],"descending":[]}#residuals from the fittings
    
    residuals["ascending"].append(frames["ascending"]["ifg_aps"]-deformation_from_0(frames["ascending"]["day"],*fits["ascending"][0][0]))
    residuals["ascending"].append(frames["ascending"]["ifg_aps"]-deformation_from_v1(frames["ascending"]["day"],*fits["ascending"][1][0]))
    
    residuals["descending"].append(frames["descending"]["ifg_aps"]-deformation_from_0(frames["descending"]["day"],*fits["descending"][0][0]))
    residuals["descending"].append(frames["descending"]["ifg_aps"]-deformation_from_v1(frames["descending"]["day"],*fits["descending"][1][0]))

    stdev=[]
    trackDirec=("ascending","descending")#the ascending and descending track
    
    for i in range(len(trackDirec)):
            direc=trackDirec[i]#Current direction
            for fitToInitial in (0,1):
                stdev.append(np.std(residuals[direc][fitToInitial]))
    
    if plot:
        #print("plot")

        #A linspace to plot the fit
        timeAxis=np.linspace(np.min(frames["ascending"]["day"]),np.max(frames["ascending"]["day"]),1000)
        
        #Create figure for fits and resudals
        fig, ax = plt.subplots(4,2,figsize=(16, 16))
        
        #setting up parameters ready for the loop
        
        labels=("Fit with no initial displacement","Fit with initial displacement")#The label for the graphs of the fits

        print(len(frames["ascending"]["ifg_aps"]))
        print(len(frames["descending"]["ifg_aps"]))
        
        minOf={"ascending":min(frames["ascending"]["ifg_aps"]),"descending":min(frames["descending"]["ifg_aps"])}#The minimum of the timeseries to draw the verticle lines
        maxOf={"ascending":max(frames["ascending"]["ifg_aps"]),"descending":max(frames["descending"]["ifg_aps"])}
        
        onsetTimes=[882,685]#The days of the EQ and the onset predicted by Lui et al., for the veritlae lines below
        
        for i in range(len(trackDirec)):
            direc=trackDirec[i]#Current direction
            for fitToInitial in (0,1):
                
                #Plot Deformation
                
                axIndex = i*2+fitToInitial#Which index is this graph in ax
                fitLabel = labels[fitToInitial]#The label for the fit
                
                ax[axIndex][0].set_title("Average Deformation over Socompa for Ascending Track")
                ax[axIndex][0].set_xlabel("Time since First Epoc (days)")
                ax[axIndex][0].set_ylabel("LOS Displacement (mm)?")
                ax[axIndex][0].plot(frames[direc]["day"],frames[direc]["ifg_aps"],".")
                if not fitToInitial:#Does this fit have no initial deforation or (else) does it have an initial deformation
                    ax[axIndex][0].plot(timeAxis,deformation_from_0(timeAxis,*fits[direc][fitToInitial][0]),label=fitLabel)
                else:
                    ax[axIndex][0].plot(timeAxis,deformation_from_v1(timeAxis,*fits[direc][fitToInitial][0]),label=fitLabel)
                ax[axIndex][0].vlines(onsetTimes[0],minOf[direc],maxOf[direc],"c","dashed", label="M6.8 Earthquake")
                ax[axIndex][0].vlines(onsetTimes[1],minOf[direc],maxOf[direc],"g","dashed", label="Liu et al. onset time")
                ax[axIndex][0].legend()       
                
                #Plot Residuals
                
                x=frames[direc]["day"]
                y=residuals[direc][fitToInitial]
                ax[axIndex][1].set_title(f"Residual - Standard Deviation {np.std(y)}")
                ax[axIndex][1].set_xlabel("Time since First Epoc (days)")
                ax[axIndex][1].set_ylabel("Residual (mm)?")
                ax[axIndex][1].plot(x,y, "dimgray")
                ax[axIndex][1].fill_between(x, y, where=(y > 0), color='blue', alpha=0.5)
                ax[axIndex][1].fill_between(x, y, where=(y < 0), color='red', alpha=0.5)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    fitOnsets=[]
    fitGrad=[]
    for track in ("ascending","descending"):
        for initialGrad in (0,1):
            fitOnsets.append((fits[track][initialGrad][0][0],errors[track][initialGrad][0]))
            fitGrad.append((fits[track][initialGrad][0][1]*365, errors[track][initialGrad][1]*365))
    #print(f"Onset time: {fits['ascending'][0][0][0]:.2f} \u00B1 {errors['ascending'][0][0]:.2f} Days")
    #print(f"Onset time: {fits['ascending'][1][0][0]:.2f} \u00B1 {errors['ascending'][1][0]:.2f} Days")
    #print(f"Onset time: {fits['descending'][0][0][0]:.2f} \u00B1 {errors['descending'][0][0]:.2f} Days")
    
    return fitOnsets, fitGrad, stdev

def getFitParamsForAverage(ascendingFrame, descendingFrame, radius, socompaLocation=(-24.3959, -68.245997),plot=False):
    frames={"ascending":get_average_deformation(ascendingFrame, radius, socompaLocation ),"descending":get_average_deformation(descendingFrame, radius, socompaLocation, ascending=False)[0]} #the ascending and descending frame
    if plot:
        map=folium.Map(location=socompaLocation, control_scale=True)
            
        folium.Circle(socompaLocation, radius*1000, color='black').add_to(map)
        map.save("map.html")
        display(IFrame('map.html', width=700, height=500))

    numOfPoints=frames["ascending"][1]
    frames["ascending"]=frames["ascending"][0]
    frames["ascending"]["ifg_aps"]=frames["ascending"]["ifg_aps"][0]
    frames["descending"]["ifg_aps"]=frames["descending"]["ifg_aps"][0]

    fitOnsets, fitGrad, stdev = getFitParams(frames["ascending"], frames["descending"], plot=plot)
    return fitOnsets, fitGrad, stdev, numOfPoints

def haveresine(reference_lat,reference_lon,centre_lat,centre_lon):
    # Radius of the Earth
    r=6371.0

    # Convert coords. to rad. from deg.
    lat2=np.radians(reference_lat)
    lat1=np.radians(centre_lat)
    lon2=np.radians(reference_lon)
    lon1=np.radians(centre_lon)

    #Find the distance between two points
    first = np.sin((lat2-lat1)/2)**2
    secondCos = np.cos(lat2)*np.cos(lat1)
    secondSin = np.sin((lon2-lon1)/2)**2
    return 2*r*np.arcsin(np.sqrt(first+secondCos*secondSin))

def cumulitive_displacement(frame, startDay=0, endDay=-1, averageOver=0):
    #find index of onset
    indexStart=[0,0]
    indexEnd=[0,0]

    
    indexStart[0]=find_index_of_closest(frame["day"],startDay)
    indexStart[1]=indexStart[0]+averageOver
    
    if endDay==-1:
        indexEnd[1]=-1
    else:
        indexEnd[1]=find_index_of_closest(frame["day"],endDay)
    indexEnd[0]=indexEnd[1]-averageOver

    #Find average displacement at start
    startAv=np.average(frame["ifg_aps"][indexStart[0]+1:indexStart[1]])
    
    #Find average displacement at end
    if endDay==-1:
        endAv=np.average(frame["ifg_aps"][indexEnd[0]+1:])
    else:
        endAv=np.average(frame["ifg_aps"][indexEnd[0]+1:indexEnd[1]+1])
        
    #Subtract

    #print(endAv)
    #print(startAv)
    
    return endAv-startAv

def find_index_of_closest(list,value,returnDist=False):
    list=np.array(list)
    diff=list-value
    index=np.argmin(abs(diff))
    
    if returnDist:
        return int(index), diff[index]

    else:
        return int(index)
    
def plot_heat_map(InSAR_Data, value_of_pixels, name_of_value, title, size_of_dot=0.1, gap_to_edge=0.05):
    # Set the region for the plot to be slightly larger than the data bounds.
    region = [
        np.min(InSAR_Data["lon"]) - gap_to_edge,
        np.max(InSAR_Data["lon"]) + gap_to_edge,
        np.min(InSAR_Data["lat"]) - gap_to_edge,
        np.max(InSAR_Data["lat"]) + gap_to_edge,
        ]

    #print(region)
    #print(data.head())

    sizes=np.ones(len(value_of_pixels))*size_of_dot

    # Create the figure
    fig = pygmt.Figure()

    # Download and plot the DEM data
    fig.grdimage(
        frame=["a",f"+t{title}"],
        region=region,
        projection="M15c",
        grid="@earth_relief_03s",  # Using 3 arc-minute resolution DEM
        shading=True,
        cmap="geo"
    )

    # Add coastlines for reference
    fig.coast(
        borders=["1/1p,yellow", "2/0.5p,yellow"],
        region=region,
        projection="M15c",
        shorelines=True,
        frame=True
    )

    # Create a color palette for earthquake depths
    pygmt.makecpt(cmap="viridis", series=[np.min(value_of_pixels), np.max(value_of_pixels)])

    # Plot the earthquake data
    fig.plot(
        x=InSAR_Data["lon"],
        y=InSAR_Data["lat"],
        size=sizes,
        fill=value_of_pixels,
        cmap=True,
        style="cc",
        pen="black",
    )

    # Add a color bar for depth
    fig.colorbar(frame=f"xaf+l{name_of_value}")

    # Show the map
    fig.show()
