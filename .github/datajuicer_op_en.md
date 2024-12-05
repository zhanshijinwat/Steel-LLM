- Data Juicer Operators Used for Text Processing

| Operator                       | Description                                                                                                  |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------|
|chinese_convert_mapper|Converts Chinese between Traditional Chinese, Simplified Chinese and Japanese Kanji|
|clean_email_mapper|Removes email information|
|clean_html_mapper|Removes HTML tags and returns plain text of all the nodes|
|clean_ip_mapper|Removes IP addresses|
|clean_links_mapper|Removes links, such as those starting with http or ftp|
|clean_copyright_mapper|Removes copyright notice at the beginning of code files (must contain the word copyright)|
|expand_macro_mapper|Expands macros usually defined at the top of TeX documents|
|fix_unicode_mapper|Fixes broken Unicodes|
|punctuation_normalization_mapper|Normalizes various Unicode punctuations to their ASCII equivalents|
|remove_repeat_sentences_mapper|Remove repeat sentences in text samples.|
|remove_specific_chars_mapper|Removes any user-specified characters or substrings|
|whitespace_normalization_mapper|Normalizes various Unicode whitespaces to the normal ASCII space (U+0020)|
|alphanumeric_filter|Keeps samples with alphanumeric ratio within the specified range|
|average_line_length_filter|Keeps samples with average line length within the specified range|
|character_repetition_filter|Keeps samples with char-level n-gram repetition ratio within the specified range|
|maximum_line_length_filter|Keeps samples with maximum line length within the specified range|
|perplexity_filter|Keeps samples with perplexity score below the specified threshold|
|special_characters_filter|Keeps samples with special-char ratio within the specified range|
|text_length_filter|Keeps samples with total text length within the specified range|
|word_repetition_filter|Keeps samples with word-level n-gram repetition ratio within the specified range|
|document_simhash_deduplicator|Deduplicates samples at document-level using SimHash|

<br></br>
- Data Juicer Operators Used for Code Processing

| Operator                       | Description                                                                                                  |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------|
|operator|description|
|clean_copyright_mapper|Removes copyright notice at the beginning of code files (must contain the word copyright)|
|clean_email_mapper|Removes email information|
|clean_links_mapper|Removes links, such as those starting with http or ftp|
|fix_unicode_mapper|Fixes broken Unicodes|
|punctuation_normalization_mapper|Normalizes various Unicode punctuations to their ASCII equivalents|
|alphanumeric_filter|Keeps samples with alphanumeric ratio within the specified range|
|average_line_length_filter|Keeps samples with average line length within the specified range|
|character_repetition_filter|Keeps samples with char-level n-gram repetition ratio within the specified range|
|maximum_line_length_filter|Keeps samples with maximum line length within the specified range|
|text_length_filter|Keeps samples with total text length within the specified range|
|words_num_filter|Keeps samples with word count within the specified range|
|word_repetition_filter|Keeps samples with word-level n-gram repetition ratio within the specified range|
|document_simhash_deduplicator|Deduplicates samples at document-level using SimHash|
