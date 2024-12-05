- 文本处理使用的data juicer算子
 
|算子|描述|
|:----|:----|
|chinese_convert_mapper|用于在繁体中文、简体中文和日文汉字之间进行转换（借助 opencc）|
|clean_email_mapper|删除邮箱信息|
|clean_html_mapper|删除 HTML 标签并返回所有节点的纯文本|
|clean_ip_mapper|删除 IP 地址|
|clean_links_mapper|删除链接，例如以 http 或 ftp 开头的|
|clean_copyright_mapper|删除代码文件开头的版权声明 (:warning: 必须包含单词 copyright)|
|expand_macro_mapper|扩展通常在 TeX 文档顶部定义的宏|
|fix_unicode_mapper|修复损坏的 Unicode（借助 ftfy）|
|punctuation_normalization_mapper|将各种 Unicode 标点符号标准化为其 ASCII 等效项|
|remove_repeat_sentences_mapper|删除样本中的重复句子|
|remove_specific_chars_mapper|删除样本中的特殊字符（用户自定义）|
|whitespace_normalization_mapper|将各类空格归一转换为英语空格|
|alphanumeric_filter|保留字母数字比例在指定范围内的样本|
|average_line_length_filter|保留平均行长度在指定范围内的样本|
|character_repetition_filter|保留 char-level n-gram 重复比率在指定范围内的样本|
|maximum_line_length_filter|保留最大行长度在指定范围内的样本|
|perplexity_filter|保留困惑度低于指定阈值的样本|
|special_characters_filter|保留 special-char 比率的在指定范围内的样本|
|text_length_filter|保留总文本长度在指定范围内的样本|
|word_repetition_filter|保留 word-level n-gram 重复比率在指定范围内的样本|
|document_simhash_deduplicator|使用 SimHash 在文档级别对样本去重|


<br></br>
- 代码处理使用的data juicer算子
  

|算子|描述|
|:----|:----|
|clean_copyright_mapper|删除代码文件开头的版权声明 (:warning: 必须包含单词 copyright)|
|clean_email_mapper|删除邮箱信息|
|clean_links_mapper|删除链接，例如以 http 或 ftp 开头的|
|fix_unicode_mapper|修复损坏的 Unicode（借助 ftfy）|
|punctuation_normalization_mapper|将各种 Unicode 标点符号标准化为其 ASCII 等效项|
|alphanumeric_filter|保留字母数字比例在指定范围内的样本|
|average_line_length_filter|保留平均行长度在指定范围内的样本|
|character_repetition_filter|保留 char-level n-gram 重复比率在指定范围内的样本|
|maximum_line_length_filter|保留最大行长度在指定范围内的样本|
|text_length_filter|保留总文本长度在指定范围内的样本|
|word_num_filter|保留字数在指定范围内的样本|
|word_repetition_filter|保留 word-level n-gram 重复比率在指定范围内的样本|
|document_simhash_deduplicator|使用 SimHash 在文档级别对样本去重|
