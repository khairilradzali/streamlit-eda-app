# pages/6_References.py
import streamlit as st
from utils import render_footer  # optional footer function

st.set_page_config(page_title="References", layout="wide")

st.markdown("---")
st.header("References")

with st.expander("Core Machine Learning Textbooks"):
    st.markdown("""
1. **Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer.**  
   🔗 https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/

2. **Hastie, T., Tibshirani, & Friedman (2009). _The Elements of Statistical Learning_.**  
   🔗 https://hastie.su.domains/ElemStatLearn/

3. **Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_.**  
   🔗 https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/
""")

with st.expander("Algorithm-Specific Research Papers"):
    st.markdown("""
4. **Breiman, L. (2001). Random Forests. _Machine Learning_.**  
   🔗 https://link.springer.com/article/10.1023/A:1010933404324

5. **Chen, T., & Guestrin, C. (2016). XGBoost. KDD.**  
   🔗 https://arxiv.org/abs/1603.02754

6. **Ke, G. et al. (2017). LightGBM. NeurIPS.**  
   🔗 https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

7. **Prokhorenkova, L. et al. (2018). CatBoost. NeurIPS.**  
   🔗 https://arxiv.org/abs/1706.09516

8. **Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE.**  
   🔗 https://ieeexplore.ieee.org/document/1053964

9. **Cortes, C., & Vapnik, V. (1995). Support Vector Networks.**  
   🔗 https://link.springer.com/article/10.1007/BF00994018
""")

with st.expander("Time Series Forecasting & Reinforcement Learning"):
    st.markdown("""
10. **Hyndman, R. & Athanasopoulos (2018). _Forecasting: Principles and Practice_.**  
    🔗 https://otexts.com/fpp3/

11. **Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_.**  
    🔗 http://incompleteideas.net/book/the-book.html
""")

# Footer
render_footer()  # optional if you have it in utils.py
