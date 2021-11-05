from tkinter import filedialog
from tkinter import messagebox
import tkinter
import os, shutil, sys, traceback

warband_fx = True
leave_comments = False #Only for /* */ comments.

directories = dict() #You can modify directories values.
directories["original dir"] = "./_Original"
directories["beautified file"] = "./_Beautified/{name}{extension}"
if warband_fx: directories["wb fx comment"] = "./warband fx comment.txt"

class Beautifier():
    def __init__(self, file):
        self.file = file
        with open(self.file, "r") as f:
            self.text = f.read()
        to_space = ["\t", "\\\n"] #Replace to space.
        for char in to_space:
            self.text = self.text.replace(char, " ")
        self.text = "\n".join([line.split("//")[0].strip(" ") for line in self.text.split("\n")]) #Remove // comments and strip.
        if not leave_comments: #Remove /* */ comments.
            result = ""
            i = 0
            commented_out = False
            while i < len(self.text):
                char = self.text[i]
                double_chars = ["/*", "*/"]
                for first, last in double_chars:
                    if char == first and i + 1 < len(self.text) and self.text[i + 1] == last:
                        char = first + last
                        i += 1
                if char in ["/*"]:
                    commented_out = True
                if not commented_out:
                    result += char
                if char in ["*/"]:
                    commented_out = False
                i += 1
            self.text = result
        processes = { #Transfer to raw, 1 line format.
            "normal" : self.raw_normal,
            "macro" : self.raw_macro
        }
        processed_lines = []
        lines = self.text.split("\n")
        for line in lines:
            if line: processed_lines.append(processes["macro" if line[0] == "#" else "normal"](line))
        self.text = "".join(processed_lines)
        to_remove_repetition = [" ", "\n"]
        for char in to_remove_repetition:
            while char*2 in self.text:
                self.text = self.text.replace(char*2, char)
        self.text = self.text.strip("\n")
        processes = { #Beautify
            "normal" : self.beautify_normal,
            "macro" : self.beautify_macro
        }
        lines = self.text.split("\n")
        processed_lines = []
        self.indent = 0
        self.macro = False
        self.commented_out = False
        self.new_line_after_comment = False
        for line in lines:
            processed_lines.append(processes["macro" if not self.commented_out and line[0] == "#" else "normal"](line))
        self.text = "".join(processed_lines)[:-1]#
    
    def write(self, name, extension):
        if warband_fx and extension == ".fx":
            with open(directories["wb fx comment"], "r") as f:
                warband_readme = f.read()
            self.text = warband_readme + self.text
        with open(directories["beautified file"].format(name = name, extension = extension), "w") as f:
            f.write(self.text)
    
    def raw_normal(self, line):
        to_split = [
            "{", "}", "[", "]", "(", ")",
            ",", ";", ".", ":",
            "=", "+", "-", "*", "/", "%",
            "?", "!",
            "&", "^", "|", "<", ">", "~"
            "\"", "'"
        ]
        splitted_line = []
        part = ""
        for char in line:
            if char in to_split:
                splitted_line.extend([part.strip(" "), char])
                part = ""
            else:
                part += char
        splitted_line.append(part.strip(" "))
        line = "".join(splitted_line)
        line += " " if line in ["else"] else ""
        return line
    
    def raw_macro(self, line):
        splitted_line = line[1:].split(" ")
        operation = splitted_line[0]
        parameters = splitted_line[1:]
        if operation in ["if", "elif", "pragma"]:
            line = "#{} {}".format(operation, self.raw_normal(" ".join(parameters)))
        elif operation in ["define"]:
            if "(" in parameters[0]:
                parameters = " ".join(parameters)
                i = parameters.index("(")
                identifier = parameters[:i]
                i += 1
                params = ""
                while i < len(parameters):
                    char = parameters[i]
                    i += 1
                    if char == ")":
                        break
                    params += char
                token_string = parameters[i + 1:]
                line = "#{} {}({}) {}".format(operation, identifier, self.raw_normal(params), self.raw_normal(token_string))
            else:
                identifier = parameters[0]
                token_string = " ".join(parameters[1:])
                line = "#{} {} {}".format(operation, identifier, self.raw_normal(token_string))
        return "\n" + line + "\n"
    
    def newline(self):
        return ("\\" if self.macro else "") + "\n" + "\t" * self.indent
    
    def beautify_normal(self, line):
        result = "\t" * self.indent if not self.macro else ""
        i = 0
        newline = False
        define_ = False
        for_loop = False
        while i < len(line):
            if newline:
                result += self.newline()
                newline = False
            char = line[i]
            if char in ["<", ">", "+", "-", "&", "|", ":"] and i + 1 < len(line) and line[i + 1] == char:
                char = char * 2
                i += 1
            double_chars = ["};", "/*", "*/"]
            for first, last in double_chars:
                if char == first and i + 1 < len(line) and line[i + 1] == last:
                    char = first + last
                    i += 1
            if warband_fx and i + 6 < len(line) and line[i : i + 7] == "DEFINE_": #For Warband .fx files only.
                define_ = True
            if i + 3 < len(line) and line[i : i + 4] == "for(":
                for_loop = True
            if char in ["*/"]:
                self.commented_out = False
                if self.new_line_after_comment:
                    self.new_line_after_comment = False
                    newline = True
            if self.commented_out:
                result += char
            elif char in ["=", "!", "<", ">", "+", "-", "*", "/", "%", "&", "^", "|", "<<", ">>"] and i + 1 < len(line) and line[i + 1] == "=":
                char = char + "="
                i += 1
                result += " " + char + " "
            elif char in ["-"] and result[-1] == " ":
                result += char
            elif char in ["+", "-", "/", "*", "%", ">", "<", "&&", "||", "&", "|", "^", "<<", ">>", "?", "="]:
                result += " " + char + " "
            elif char in [",", ":"]:
                result += char + " "
            elif char in [";"]:
                result += char + (" " if for_loop else "")
                if not for_loop:
                    newline = True
            elif char in ["{"]:
                self.indent += 1
                result += char
                newline = True
            elif char in ["}", "};"]:
                self.indent -= 1
                if result[-1] == "\t":
                    result = result[:-1]
                else:
                    result += self.newline()
                result += char
                newline = True
            elif char in [")"] and define_:
                define_ = False
                result += char
                newline = True
            elif char in [")"] and for_loop:
                for_loop = False
                result += char
            elif char in ["/*"]:
                self.commented_out = True
                if (not result) or result[-1] == "\t":
                    self.new_line_after_comment = True
                result += char
            else:
                result += char
            i += 1
        return result + ("\n" if not self.macro else "")
    
    def beautify_macro(self, line):
        try_begin_ops = ["if", "ifdef", "ifndef"]
        else_try_ops = ["elif", "else"]
        try_end_ops = ["endif"]
        splitted_line = line[1:].split(" ")
        operation = splitted_line[0]
        parameters = splitted_line[1:]
        if operation in try_end_ops + else_try_ops:
            self.indent -= 1
        self.macro = True
        if operation in ["if", "elif", "pragma"]:
            line = "#{} {}".format(operation, self.beautify_normal(" ".join(parameters)))
        elif operation in ["define"]:
            if "(" in parameters[0]:
                parameters = " ".join(parameters)
                i = parameters.index("(")
                identifier = parameters[:i]
                i += 1
                params = ""
                while i < len(parameters):
                    char = parameters[i]
                    i += 1
                    if char == ")":
                        break
                    params += char
                self.indent += 1
                token_string = self.beautify_normal(parameters[i + 1:])
                line = "#{} {}({}) {}{}".format(operation, identifier, self.beautify_normal(params), ("\\\n" + self.indent * "\t") if len(token_string.split("\n")) > 1 else "", token_string)
                self.indent -= 1
            else:
                identifier = parameters[0]
                token_string = " ".join(parameters[1:])
                line = "#{} {} {}".format(operation, identifier, self.beautify_normal(token_string))
        self.macro = False
        result = "\t" * self.indent + line + "\n"
        if operation in try_begin_ops + else_try_ops:
            self.indent += 1
        return result

tkinter.Tk().withdraw()
file = tkinter.filedialog.askopenfilename(title = "Select a File", initialdir = directories["original dir"], filetypes = (("all files", "*.*"),))
basename = os.path.basename(file)
name, extension = os.path.splitext(basename)
if not name:
    messagebox.showerror("Error", "Process canceled.")
    sys.exit()
try:
    beautifier = Beautifier(file)
    beautifier.write(name, extension)
except:
    messagebox.showerror("Error", traceback.format_exc())
    sys.exit()
beautified_file_basename = os.path.basename(directories["beautified file"]).format(name = name, extension = extension)
messagebox.showinfo("Process Complete", "\"{}\" is processed successfuly".format(basename)
                    + (" as \"{}\"".format(beautified_file_basename) if beautified_file_basename != basename else "")
                    + ".")
