/* Run this as root: $mysql -u root -p < DB_CONTRACTS_CLEAR.sql */

/* Drop databases */
DROP DATABASE IF EXISTS STIR_CME_C;
DROP DATABASE IF EXISTS STIR_CME_W;
DROP DATABASE IF EXISTS STIR_CME_S;
DROP DATABASE IF EXISTS STIR_CME_SM;
DROP DATABASE IF EXISTS STIR_CME_BO;
DROP DATABASE IF EXISTS STIR_CME_FC;
DROP DATABASE IF EXISTS STIR_CME_LC;
DROP DATABASE IF EXISTS STIR_CME_CL;
DROP DATABASE IF EXISTS STIR_CME_HO;
DROP DATABASE IF EXISTS STIR_CME_NG;
DROP DATABASE IF EXISTS STIR_ICE_B;
DROP DATABASE IF EXISTS STIR_ICE_G;
DROP DATABASE IF EXISTS STIR_CME_EH;
DROP DATABASE IF EXISTS STIR_CME_RB;
DROP DATABASE IF EXISTS STIR_CME_GC;
DROP DATABASE IF EXISTS STIR_CME_HG;
DROP DATABASE IF EXISTS STIR_CME_PA;
DROP DATABASE IF EXISTS STIR_CME_PL;
DROP DATABASE IF EXISTS STIR_CME_SI;
DROP DATABASE IF EXISTS STIR_ICE_CT;
DROP DATABASE IF EXISTS STIR_ICE_CC;
DROP DATABASE IF EXISTS STIR_ICE_KC;
DROP DATABASE IF EXISTS STIR_ICE_OJ;
DROP DATABASE IF EXISTS STIR_ICE_SB;
DROP DATABASE IF EXISTS STIR_CME_RR;
DROP DATABASE IF EXISTS STIR_ICE_O;
DROP DATABASE IF EXISTS STIR_ICE_CO;

/* Create additional database */
CREATE DATABASE IF NOT EXISTS STIR_CME_C;
CREATE DATABASE IF NOT EXISTS STIR_CME_W;
CREATE DATABASE IF NOT EXISTS STIR_CME_S;
CREATE DATABASE IF NOT EXISTS STIR_CME_SM;
CREATE DATABASE IF NOT EXISTS STIR_CME_BO;
CREATE DATABASE IF NOT EXISTS STIR_CME_FC;
CREATE DATABASE IF NOT EXISTS STIR_CME_LC;
CREATE DATABASE IF NOT EXISTS STIR_CME_CL;
CREATE DATABASE IF NOT EXISTS STIR_CME_HO;
CREATE DATABASE IF NOT EXISTS STIR_CME_NG;
CREATE DATABASE IF NOT EXISTS STIR_ICE_B;
CREATE DATABASE IF NOT EXISTS STIR_ICE_G;
CREATE DATABASE IF NOT EXISTS STIR_CME_EH;
CREATE DATABASE IF NOT EXISTS STIR_CME_RB;
CREATE DATABASE IF NOT EXISTS STIR_CME_GC;
CREATE DATABASE IF NOT EXISTS STIR_CME_HG;
CREATE DATABASE IF NOT EXISTS STIR_CME_PA;
CREATE DATABASE IF NOT EXISTS STIR_CME_PL;
CREATE DATABASE IF NOT EXISTS STIR_CME_SI;
CREATE DATABASE IF NOT EXISTS STIR_ICE_CT;
CREATE DATABASE IF NOT EXISTS STIR_ICE_CC;
CREATE DATABASE IF NOT EXISTS STIR_ICE_KC;
CREATE DATABASE IF NOT EXISTS STIR_ICE_OJ;
CREATE DATABASE IF NOT EXISTS STIR_ICE_SB;
CREATE DATABASE IF NOT EXISTS STIR_CME_RR;
CREATE DATABASE IF NOT EXISTS STIR_ICE_O;
CREATE DATABASE IF NOT EXISTS STIR_ICE_CO;

/* Create user */
GRANT ALL PRIVILEGES ON STIR.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_C.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_W.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_S.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_SM.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_BO.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_FC.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_LC.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_CL.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_HO.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_NG.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_B.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_G.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_EH.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_RB.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_GC.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_HG.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_PA.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_PL.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_SI.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_CT.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_CC.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_KC.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_OJ.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_SB.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_CME_RR.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_O.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON STIR_ICE_CO.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;

FLUSH PRIVILEGES;
