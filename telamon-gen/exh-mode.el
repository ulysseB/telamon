;; exh.el - A major mode for Telamon EXH files
;;
;; Author: Andi Drebes <andi@drebesium.org>
;;
;; Based on the Mode Tutorial [1] and Sample Mode [2] from
;; emacswiki.org.
;; 
;; [1] https://www.emacswiki.org/emacs/ModeTutorial
;; [2] https://www.emacswiki.org/emacs/SampleMode
;;

;;; Key map; empty by default
(defvar exh-mode-map
  (let ((map (make-sparse-keymap)))
    map)
  "Keymap for `exh-mode'.")

;;; Syntax highlighting
(defvar exh-mode-syntax-table 
  (let ((st (make-syntax-table)))
    (modify-syntax-entry ?\/ ". 12b" st)
    (modify-syntax-entry ?\n "> b" st) st)
  "Syntax table for `exh-mode'.")

;; Optimized regexp generated with regexp-opt:
;; (regexp-opt '("define" "require" "requires" "value" "on_change" "alias" "on_set" "on_unset" "enum" "at" "most" "sum" "counter" "symmetric" "product" "getter" "end" "forall" "in" "antisymmetric" "external" "set" "subsetof" "mul" "trigger" "when" "quotient" "of" "forall" "in" "is" "not") t)

(defconst exh-font-lock-keywords
  (list
   '("\\_<\\(a\\(?:lias\\|ntisymmetric\\|t\\)\\|counter\\|define\\|e\\(?:n\\(?:d\\|um\\)\\|xternal\\)\\|forall\\|getter\\|i[ns]\\|m\\(?:ost\\|ul\\)\\|not\\|o\\(?:f\\|n_\\(?:change\\|\\(?:un\\)?set\\)\\)\\|product\\|quotient\\|requires?\\|s\\(?:et\\|u\\(?:bsetof\\|m\\)\\|ymmetric\\)\\|trigger\\|value\\|when\\|include\\)\\_>" . font-lock-keyword-face)
   '("\\$\\([a-zA-Z0-9_-]*\\)" . font-lock-variable-name-face))
  "Highlighting for EXH mode")

;;; Indentation
(defvar default-tab-width 2)

(defun exh-indent-line ()
  "Indent current line as EXH code"
  (interactive)
  (beginning-of-line)
  (if (bobp) ; Indent to 0 if beginning of buffer
      (indent-line-to 0)

    (let ((not-indented t) cur-indent)

       ; If point is in front of "end" -> decrease indentation
      (if (looking-at "^[ \t]*end")
          (progn
            (save-excursion
              (forward-line -1)
              (setq cur-indent (- (current-indentation) default-tab-width)))
	    (if (< cur-indent 0)
                (setq cur-indent 0)))
	(save-excursion 
          (while not-indented
            (forward-line -1)

	    ; Use same indentation as previous "end"
            (if (looking-at "^[ \t]*end")
                (progn
                  (setq cur-indent (current-indentation))
                  (setq not-indented nil))

	       ; Increase indentation upon "define", "quotient" or "set"
              (if (looking-at "^[ \t]*\\(define\\|quotient\\|set\\)")
                  (progn
                    (setq cur-indent (+ (current-indentation) default-tab-width))
                    (setq not-indented nil))

		; Beginning of buffer reached
                (if (bobp)
                    (setq not-indented nil)))))))
      (if cur-indent
          (indent-line-to cur-indent)
        (indent-line-to 0)))))

;;;###autoload
(define-derived-mode exh-mode fundamental-mode "EXH"
  "A major mode for editing EXH files."
  :syntax-table exh-mode-syntax-table
  (setq-local font-lock-defaults
	      '(exh-font-lock-keywords))
  (setq-local indent-line-function 'exh-indent-line))

(provide 'exh-mode)
