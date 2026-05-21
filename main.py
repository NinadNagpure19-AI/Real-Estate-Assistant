import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False

# Sidebar URL inputs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("Process URLs")

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]

    if not urls:
        st.warning("Please provide at least one valid URL.")
    else:
        status_box = st.empty()
        try:
            for status in process_urls(urls):
                status_box.text(status)
            st.session_state.processed = True
            st.success("Ready! Ask a question below.")
        except Exception as e:
            st.error(f"Error during processing: {e}")

# Question input
query = st.text_input("Question")

if query:
    if not st.session_state.processed:
        st.warning("Please process URLs first.")
    else:
        try:
            with st.spinner("Generating answer..."):
                answer, sources = generate_answer(query)
            st.header("Answer")
            st.write(answer)
            if sources:
                st.subheader("Sources")
                for source in sources:
                    st.write(source)
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")