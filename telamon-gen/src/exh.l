%option reentrant
%option noyywrap

%{
	int lines = 0;

	enum token {
		COMMENT,
		OTHER,
	};
%}

/* 6 Patterns § http://westes.github.io/flex/manual/Patterns.html#Patterns */
%%
^["/"{2}|"/"{3}].+   { ++lines; return COMMENT; }
[a-zA-Z][a-zA-Z0-9]* { ++lines; return OTHER; }

<<EOF>> return EOF;
%%
