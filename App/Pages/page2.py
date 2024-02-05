import streamlit as st
import fitz
import io
import re
import fitz
import pickle

contractions_dict_path = r"C:\Users\londo\01\001\Repos\Sfere\Notebooks\03-Feature-Extraction\contractions_dict.pkl"
with open(contractions_dict_path, 'rb') as file:
    contractions_dict = pickle.load(file)

def prepare_pdf_pages(pdf_buffer, skip_pages=0, skip_lines=0):
    """
    Opens a PDF document from a bytes buffer and prepares its pages for text extraction.

    :param pdf_buffer: Bytes buffer of the PDF document.
    :param skip_pages: Number of initial pages to skip.
    :param skip_lines: Number of initial lines to skip on the first processed page.
    :return: List of prepared pages for text extraction.
    """
    doc = fitz.open(stream=pdf_buffer, filetype="pdf")
    prepared_pages = []

    for page_number in range(skip_pages, len(doc)):
        page = doc[page_number]
        blocks = page.get_text("blocks")
        sorted_blocks = sorted(blocks, key=lambda block: block[1])  # Sort text blocks by vertical position (top to bottom)
        
        # Skip lines only for the first processed page
        if page_number == skip_pages:
            prepared_pages.append(sorted_blocks[skip_lines:])
        else:
            prepared_pages.append(sorted_blocks)

    return prepared_pages, doc

def format_1(prepared_pages, doc):
    try:
        text = ""
        for blocks in prepared_pages:
            for block in blocks:
                block_text = block[4].strip()
                if block_text:
                    if text and not text.endswith('.'):
                        text += ' ' + block_text
                    else:
                        text += block_text
                    if block_text.endswith('.'):
                        text += '\n\n'
        doc.close()
        return text.strip()
    except IndexError:
        return "Document not compatible with this format 🚫"

def format_2(prepared_pages):
    try:
        text = ""
        previous_block_ended_with_period = False

        for blocks in prepared_pages:
            for block in blocks:
                block_text = block[4].strip()
                if block_text:
                    if block_text[-1] in ".!?":
                        previous_block_ended_with_period = True
                    elif previous_block_ended_with_period and block_text[0].isupper():
                        text += '\n\n'
                        previous_block_ended_with_period = False

                    text += block_text + ' '

        return text.strip()
    except IndexError:
        return "Document not compatible with Format 2 🚫"

def format_3(prepared_pages):
    try:
        formatted_text = ""
        previous_block_ended_with_punctuation = False

        for page_blocks in prepared_pages:
            for i, block in enumerate(page_blocks):
                block_text = block[4].strip()
                block_ends_with_punctuation = block_text[-1] in '.!?";'

                if previous_block_ended_with_punctuation or i == 0:
                    formatted_text += "\n\n" + block_text
                else:
                    formatted_text += " " + block_text

                previous_block_ended_with_punctuation = block_ends_with_punctuation

        return formatted_text.strip()
    except IndexError:
        return "Document not compatible with Format 3 🚫"


def format_4(prepared_pages):
    try:
        formatted_text = ""
        previous_block_ended_with_punctuation = False  # Track if the previous block ended with punctuation

        for page_blocks in prepared_pages:
            for i, block in enumerate(page_blocks):
                block_text = block[4].strip()  # Extract the text from the block

                # Check if this block is a page number or header/footer which we want to ignore for paragraph formatting
                if block_text.startswith("Page") and block_text.endswith("Twain"):
                    continue

                # Determine if the current block ends with punctuation
                block_ends_with_punctuation = block_text[-1] in '.!?";'

                # If the previous block ended with punctuation, or it's the start of a document, add a paragraph break
                if previous_block_ended_with_punctuation or (formatted_text == ""):
                    formatted_text += block_text
                else:
                    # Else, just add a space and then the text
                    formatted_text += " " + block_text

                # Update for the next iteration
                previous_block_ended_with_punctuation = block_ends_with_punctuation

                # Add a line break if the block ends with punctuation, otherwise just a space
                formatted_text += "\n\n" if block_ends_with_punctuation else " "

        return formatted_text.strip()
    except IndexError:
        return "Document not compatible with Format 4 🚫"



def count_pages_and_lines(pdf_buffer):
    """
    Counts the number of pages and total lines in a PDF document.

    :param pdf_file: File-like object of the PDF document.
    :return: A tuple containing the total number of pages and total number of lines.
    """
    with fitz.open("pdf", pdf_buffer) as doc:
        total_pages = len(doc)
        total_lines = 0

        for page in doc:
            text = page.get_text("text")
            lines = text.split('\n')
            total_lines += len(lines)
    
    return total_pages, total_lines 

def remove_links(text):
    """
    Removes words that start with 'http', 'www.', or end with domain extensions like '.com', '.co.uk', '.org', etc.

    :param text: Text to process.
    :return: Text with hyperlinks removed.
    """
    # Regular expression for matching URLs
    # This pattern matches words that start with 'http' or 'www.' and words that end with common domain extensions
    url_pattern = r'\b(https?://\S+|www\.\S+|\S+\.(com|org|net|co\.uk|edu|gov|mil|info))\b'

    # Remove URLs
    text_without_links = re.sub(url_pattern, '', text)

    return text_without_links

def remove_chapter_headers(text):
    # This regular expression matches "Chapter" or "CHAPTER" followed by a space and any word
    pattern = r'\b(?:CHAPTER|Chapter)\s+\S+'
    # Replace the matched patterns with an empty string
    return re.sub(pattern, '', text)

def remove_section_headers(text):
    """
    Removes Roman numerals at the start of lines if followed by a word starting with a capital letter,
    and also removes lines starting with 'PART' followed by any word.

    :param text: Text to process.
    :return: Text with specified headers removed.
    """
    # Regular expression pattern
    # This pattern matches:
    # 1. 'PART' followed by a space and any word (the entire line is removed)
    # 2. A line starting with a Roman numeral (up to 10, for simplicity) followed by a space
    #    (only the numeral and the space are removed)
    pattern = r'\bPART\s+\S+|^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\s+(?=[A-Z])'

    # Replace the matched patterns with an empty string
    # For the Roman numerals, only the numeral and following space are removed, keeping the rest of the line
    return re.sub(pattern, '', text, flags=re.MULTILINE)


def remove_page_headers_and_number_lines(text):
    """
    Removes page headers and lines that are just a number from text.

    :param text: Text to remove page headers and number-only lines from.
    :return: Text with page headers and number-only lines removed.
    """
    # The regular expression pattern:
    # - '^(Page .*)$' matches lines starting with 'Page' followed by any characters.
    # - '^\d+$' matches lines that consist only of digits.
    # 're.IGNORECASE' makes the pattern case-insensitive.
    # 're.MULTILINE' treats each line as a separate string.
    pattern = r'^(Page .*)$|^\d+$'
    return re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

def remove_all_linebreaks(text):
    """
    Replaces all linebreaks in the text with a space.

    :param text: Text to replace linebreaks in.
    :return: Text with linebreaks replaced by spaces.
    """
    # Replace all types of linebreaks with a space
    return text.replace('\n', ' ').replace('\r', ' ')

def remove_words(text, words_to_remove):
    """
    Removes all instances of specified words or phrases from the given text.

    :param text: The text from which words or phrases should be removed.
    :param words_to_remove: A list of words or phrases to be removed from the text.
    :return: The text with the specified words or phrases removed.
    """
    for word in words_to_remove:
        text = text.replace(word, '')
    return text

def validate_removal_input(word, full_text):
    """
    Validates if the given word or phrase is valid for removal.

    :param word: The word or phrase to be validated.
    :param full_text: The full text from which the word or phrase will be removed.
    :return: Tuple (is_valid, error_message). is_valid is True if validation passes, False otherwise.
    """
    if not word:
        return False, "⚠️ Please enter a word or phrase to remove."
    elif word not in full_text:
        return False, "🔍 The word or phrase to remove is not found in the document."
    return True, ""

def validate_input(original_text, replacement_text, full_text):
    if not original_text or not replacement_text:
        return False, "⚠️ Please enter both the text to be replaced and the replacement text."
    elif original_text not in full_text:
        return False, "🔍 The text to be replaced is not found in the document."
    return True, ""

# Function to replace a full sentence or word
def replace_text(text, original, replacement):
    escaped_original = re.escape(original)
    return re.sub(r'(?<!\w)({})(?!\w)'.format(escaped_original), replacement, text)

def expand_contractions(text, contractions_dict):
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        # This handles cases like "I'll" where the first character is capital
        expanded_contraction = contractions_dict.get(match.lower(), match)
        if expanded_contraction != match:
            return first_char + expanded_contraction[1:]
        return expanded_contraction

    # Pattern to match contractions. The word boundaries (\b) ensure that we're capturing whole words.
    contractions_pattern = re.compile(r'\b({})\b'.format('|'.join(map(re.escape, contractions_dict.keys()))), flags=re.IGNORECASE)
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

def remove_numbers_at_start_of_sentences(text):
    """
    Removes numbers from the start of sentences in the text.

    :param text: The text from which numbers at the start of sentences should be removed.
    :return: Text with numbers removed from the start of sentences.
    """
    # Regular expression pattern
    # This pattern matches a number at the start of the text or after a sentence-ending punctuation followed by a space
    number_pattern = r'(?:^|(?<=[.!?]\s))\d+\s+'
    
    # Replace the matched patterns with an empty string
    # This removes the number and the following space, keeping the rest of the text intact
    return re.sub(number_pattern, '', text, flags=re.MULTILINE)

def capitalize_first_word(text):
    """
    Converts strings of words in all caps to just the first word capitalized.

    :param text: Text to transform.
    :return: Text with the first word of all-caps sequences capitalized.
    """

    def replace_func(match):
        # Split the match into words
        words = match.group().split()
        # Capitalize the first word and convert the rest to lowercase
        return ' '.join([words[0].capitalize()] + [word.lower() for word in words[1:]])

    # This regular expression matches strings of uppercase words
    pattern = r'\b(?:[A-Z]+(?:\s+[A-Z]+)*)\b'
    # Replace the matched patterns using the replace_func
    return re.sub(pattern, replace_func, text)

def file_uploader():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Read the file into a buffer
        pdf_buffer = io.BytesIO(uploaded_file.read())

        # Use the buffer to get the total pages and lines
        total_pages, total_lines = count_pages_and_lines(pdf_buffer)

        # Ask the user for the number of pages and lines to skip
        pages_to_skip = st.number_input("Select number of initial pages to skip", min_value=0, value=0, max_value=total_pages)
        lines_to_skip = st.number_input("Select number of initial lines to skip", min_value=0, value=0, max_value=total_lines)

        # Prepare the PDF pages for text extraction
        prepared_pages, doc = prepare_pdf_pages(pdf_buffer, skip_pages=pages_to_skip, skip_lines=lines_to_skip)

        # Select the formatting option
        formatting_option = st.selectbox(
            "Choose paragraph formatting option",
            ("Format 1: Joins text blocks, adding paragraphs after periods for well-defined sentence structures.",
             "Format 2: Groups text blocks into paragraphs based on punctuation and capital letters.",
             "Format 3: Enforces paragraph breaks strictly at punctuation marks and at the start of new pages.",
             "Format 4: Uses punctuation for paragraph breaks, omits automatic page breaks, and ignores headers and footers."
             ),
        )

        # Apply the selected formatting function
        if formatting_option == "Format 1: Joins text blocks, adding paragraphs after periods for well-defined sentence structures.":
            text = format_1(prepared_pages, doc)
        elif formatting_option == "Format 2: Groups text blocks into paragraphs based on punctuation and capital letters.":
            text = format_2(prepared_pages)
        elif formatting_option == "Format 3: Enforces paragraph breaks strictly at punctuation marks and at the start of new pages.":
            text = format_3(prepared_pages)
        else:
            text = format_4(prepared_pages)
        
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        row3_col1, row3_col2, row3_col3 = st.columns(3)
        row4_col1, row4_col2, row4_col3 = st.columns(3)
        
        
        with row1_col1:
            checkbox1 = st.checkbox("Remove Links")
            if checkbox1:
                text = remove_links(text)

        with row1_col2:
            checkbox2 = st.checkbox('Remove Chapter Headers')
            if checkbox2:
                text = remove_chapter_headers(text)
                
        with row1_col3:
            checkbox3 = st.checkbox('Expand Contractions')
            if checkbox3:
                text = expand_contractions(text, contractions_dict)
                        
                        
        with row2_col1:
            checkbox4 = st.checkbox('Remove Page Headers')
            if checkbox4:
                text = remove_page_headers_and_number_lines(text)

        with row2_col2:
            checkbox5 = st.checkbox('Remove All Linebreaks')
            if checkbox5:
                text = remove_all_linebreaks(text)
                
        with row2_col3:
            checkbox6 = st.checkbox('Remove Section Headers')
            if checkbox6:
                text = remove_section_headers(text)
                
        with row3_col1:
            checkbox7 = st.checkbox("Remove Numbers at Start of Sentences")
            if checkbox7:
                text = remove_numbers_at_start_of_sentences(text)

        with row3_col2:
            checkbox8 = st.checkbox("Remove ALL-CAPS Words")
            if checkbox8:
                text = capitalize_first_word(text)                
        
                
        with row4_col1:
            checkbox9 = st.checkbox('Remove Words')
            if checkbox9:
                # Toggle for single or multiple removals
                with st.container():
                    remove_mode = st.radio("Choose removal mode", ('Single', 'Multiple'))

                    if remove_mode == 'Single':
                        any_word = st.text_area('Word or words to remove', help='Case-sensitive')
                        if st.button('Remove Word'):
                            # Apply single removal
                            is_valid, error_message = validate_removal_input(any_word, text)
                            if not is_valid:
                                st.error(error_message)
                            else:
                                text = remove_words(text, [any_word])
                    elif remove_mode == 'Multiple':
                        # Dynamic addition of words to remove
                        num_removals = st.number_input('Number of word inputs to remove', min_value=1, value=1, step=1)
                        words_to_remove = []
                        for n in range(num_removals):
                            word_to_remove = st.text_area(f"Word/Words {n+1} to remove", key=f'remove{n+1}')
                            words_to_remove.append(word_to_remove)

                        if st.button('Apply All Removals'):
                            # Apply all removals
                            for word in words_to_remove:
                                is_valid, error_message = validate_removal_input(word, text)
                                if not is_valid:
                                    st.error(error_message)
                                    break  # Exit the loop to prevent further removals if an error occurs
                                else:
                                    text = remove_words(text, [word])
                
        with row4_col2:
                checkbox10 = st.checkbox('Replace Words')
                if checkbox10:
                    # Toggle for single or multiple replacements
                    with st.container():
                        replace_mode = st.radio("Choose replacement mode", ('Single', 'Multiple'))

                        if replace_mode == 'Single':
                            original_text = st.text_area("Enter the word or sentence to be replaced:", help="Case sensitive")
                            replacement_text = st.text_area("Enter the new word or sentence:")
                            if st.button('Apply Replacement'):
                                # Apply single replacement
                                is_valid, error_message = validate_input(original_text, replacement_text, text)
                                if not is_valid:
                                    st.error(error_message)
                                else:
                                    text = replace_text(text, original_text, replacement_text)
                        elif replace_mode == 'Multiple':
                            # Dynamic addition of input pairs
                            num_replacements = st.number_input('Number of replacements', min_value=1, value=1, step=1)
                            replacements = []
                            for n in range(num_replacements):
                                original_text = st.text_area(f"Original text {n+1}", key=f'original{n+1}')
                                replacement_text = st.text_area(f"Replacement text {n+1}", key=f'replacement{n+1}')
                                replacements.append((original_text, replacement_text))

                            if st.button('Apply All Replacements'):
                                # Apply all replacements
                                for original, replacement in replacements:
                                    # Call the validation function
                                    is_valid, error_message = validate_input(original, replacement, text)
                                    if not is_valid:
                                        st.error(error_message)
                                        break  # Exit the loop to prevent further replacements if an error occurs
                                    else:
                                        # Call the replace_text function
                                        text = replace_text(text, original, replacement)

        st.download_button('Download as .txt', text, file_name = "extracted_text.txt")
                
        st.write(text)
    else:
        st.write("Please upload a PDF file")

def show():
    st.header("Text Processor")
    st.subheader("An Interactive Tool for Efficient Document Handling")
    file_uploader()

