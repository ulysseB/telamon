" Vim syntax file
" Language: Exhaust Optimization Description
" Maintainer: Ulysse Beaugnon
" Latest Revision: 15 October 2016

if exists("b:current_syntax")
  finish
endif

syn keyword ExhStatement define require requires value on_change alias on_set on_unset enum at
syn keyword ExhStatement most sum counter symmetric product getter end forall in
syn keyword ExhStatement antisymmetric external set subsetof mul trigger when quotient of
syn keyword ExhCondition forall in is not
syn keyword ExhTodo FIXME TODO
syn match ExhComment '//.*' contains=ExhTodo
syn match ExhDoc '///.*'
syn match ExhVar '$[a-zA-Z0-9_-]*'
syn region ExhCode start='"' end='"' contains=ExhVar
syn region ExhComment start='/\*' end='\*/'

hi def link ExhStatement Statement
" hi def link ExhCondition Type
hi def link ExhComment Comment
hi def link ExhTodo Todo
hi def link ExhVar PreProc
hi def link ExhDoc PreProc
hi def link ExhCode Constant
