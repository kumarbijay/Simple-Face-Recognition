img = st.camera_input("Take a picture")
if img is not None:
    bytes_data = img.getvalue()
    st.write(bytes_data)