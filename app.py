#brew install libomp
# To run: streamlit run app.py

'''
ENTRY POINT TO RUN WEBAPP
'''
import sys
sys.path.append('src')  

from pipeline import main  

if __name__ == '__main__':
    main()  
