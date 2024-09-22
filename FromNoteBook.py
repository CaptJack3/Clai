!pip install ptan tensorboardX
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
    print(gpu_info)
print("Finish")
##
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
print("Finish")
##
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
##
from google.colab import drive
# drive.mount('/content/drive')
!git clone https://github.com/aelashkin/CLAI-HW1.git /content/CLAI
# Verify the contents
!ls /content/CLAI
import sys
sys.path.append('/content/CLAI')
##
!tar xvf /content/CLAI/data/ch08-small-quotes.tgz -C /content/CLAI/data/
##
'''  --------'''
'''  Step 2'''
'''  --------'''
# Import necessary libraries
import pandas as pd

# Define the path to the CSV file
file_path = 'CLAI/data/YNDX_160101_161231.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first 50 lines of the DataFrame
df.head(50)
##
import pandas as pd
import plotly.graph_objects as go

# Define the path to the CSV file
file_path = 'CLAI/data/YNDX_160101_161231.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert the <DATE> and <TIME> columns to a single datetime column
df['datetime'] = pd.to_datetime(df['<DATE>'].astype(str) + df['<TIME>'].astype(str), format='%Y%m%d%H%M%S')

# Select the first 50 rows for visualization
df = df.head(50)

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                     open=df['<OPEN>'],
                                     high=df['<HIGH>'],
                                     low=df['<LOW>'],
                                     close=df['<CLOSE>'])])

# Update the layout of the chart
fig.update_layout(title='Candlestick Chart for Yandex Stock Prices (First 50 Minutes of 2016)',
                  xaxis_title='Time',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

# Show the chart
fig.show()
##
import enum

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2
##
