import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
odb = session.odbs['C:/Users/and008/Documents/Models V2/M40.odb']
session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
    computeOrder=EXTRAPOLATE_AVERAGE_COMPUTE)
session.fieldReportOptions.setValues(printXYData=OFF, printTotal=OFF)
session.writeFieldReport(fileName='abaqus2.rpt', append=ON,
    sortItem='Node Label', odb=odb, step=0, frame=10, outputPosition=NODAL,
    variable=(('LE', INTEGRATION_POINT, ((INVARIANT, 'Max. Principal'), )),
    ), stepFrame=SPECIFY)