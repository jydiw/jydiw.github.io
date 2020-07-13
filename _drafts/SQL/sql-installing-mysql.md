---
title: "SQL: Installing MySQL to Practice"
excerpt_separator: "<!--more-->"
categories:
  - SQL
tags:
  -
  -
---

# Installing MySQL

1. go to https://dev.mysql.com/downloads/mysql/
2. select appropriate operating system and download
3. open the msi file
4. select the components you wish to install
5. if you are missing drivers, click "execute" for each of the requirements
6. after missing driver installation is complete, download and install mysql components by clicking "execute" again
7. configure mysql server
   1. choose "development computer"
   2. keep default ports
8. select the encrypted password
9. set password for Root account
   1.  has to have upper/lower alphanumeric, and a symbol
   2.  REMEMBER YOUR PASSWORD
10. run mysql
11. check if mysql service is running
    1.  start > services
    2.  search for mysql80
12. run mysql workbench and connect to local instance


# Installing the Employees Database

1. https://dev.mysql.com/doc/employee/en/
2. follow instructions (i git cloned)
3. instead of installing using command line, use mysql workbench
   1. https://www.youtube.com/watch?v=qHjvGAMPzEw
   2. File > Run SQL script
   3. (Wait approx 30-45 seconds)
4. go to schemas and hit refresh