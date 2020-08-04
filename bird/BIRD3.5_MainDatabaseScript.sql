--
-- PostgreSQL database dump
--

-- Dumped from database version 10.12 (Ubuntu 10.12-0ubuntu0.18.04.1)
-- Dumped by pg_dump version 12.2

-- Started on 2020-06-19 09:58:37

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 296 (class 1255 OID 23530)
-- Name: client_partner_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.client_partner_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
begin
NEW.create_date=now();
return new;
end;
$$;


ALTER FUNCTION public.client_partner_before_insert() OWNER TO postgres;

--
-- TOC entry 297 (class 1255 OID 23531)
-- Name: client_partner_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.client_partner_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=now();
if (OLD.status !='D' and NEW.status='D') THEN
NEW.client_name =CONCAT (OLD.client_name,'_', now());
NEW.client_admin_email = CONCAT(OLD.client_admin_email,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.client_partner_before_update() OWNER TO postgres;

--
-- TOC entry 298 (class 1255 OID 23532)
-- Name: cp_feature_access_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cp_feature_access_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=now();
return new;
end;
$$;


ALTER FUNCTION public.cp_feature_access_before_insert() OWNER TO postgres;

--
-- TOC entry 299 (class 1255 OID 23533)
-- Name: cp_feature_access_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cp_feature_access_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=now();
return new;
end;
$$;


ALTER FUNCTION public.cp_feature_access_before_update() OWNER TO postgres;

--
-- TOC entry 300 (class 1255 OID 23534)
-- Name: cp_group_after_update_for_updating_user_roleid(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cp_group_after_update_for_updating_user_roleid() RETURNS trigger
    LANGUAGE plpgsql
    AS $$ BEGIN if(OLD.role_id!=NEW.role_id) then update user_account ua set role_id=new.role_id where new.client_id=client_id and new.group_id::varchar=group_id::varchar; END if; return new; END; $$;


ALTER FUNCTION public.cp_group_after_update_for_updating_user_roleid() OWNER TO postgres;

--
-- TOC entry 301 (class 1255 OID 23535)
-- Name: cp_groups_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cp_groups_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cp_groups_before_insert() OWNER TO postgres;

--
-- TOC entry 302 (class 1255 OID 23536)
-- Name: cp_groups_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cp_groups_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (NEW.status ='D') THEN
NEW.group_name =CONCAT(OLD.group_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.cp_groups_before_update() OWNER TO postgres;

--
-- TOC entry 303 (class 1255 OID 23537)
-- Name: cu_connection_access_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_connection_access_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
IF (NEW.status ='D') THEN
new.connection_access_name = concat(old.connection_access_name,'_', now());
update cu_shared_connection_access set status ='D' where cu_connection_access_id= NEW.connection_access_id;
END IF;
return new;
end;
$$;


ALTER FUNCTION public.cu_connection_access_before_update() OWNER TO postgres;

--
-- TOC entry 304 (class 1255 OID 23538)
-- Name: cu_dashboard_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_dashboard_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_dashboard_before_insert() OWNER TO postgres;

--
-- TOC entry 305 (class 1255 OID 23539)
-- Name: cu_dashboard_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_dashboard_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_dashboard_before_update() OWNER TO postgres;

--
-- TOC entry 306 (class 1255 OID 23540)
-- Name: cu_report_visualization_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_report_visualization_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_report_visualization_before_insert() OWNER TO postgres;

--
-- TOC entry 307 (class 1255 OID 23541)
-- Name: cu_report_visualization_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_report_visualization_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (NEW.status ='D') THEN
NEW.report_visualization_name =CONCAT(OLD.report_visualization_name,'_', now());
update cu_alert set status='D' where report_visualization_id= old.cu_report_visualization_id ;
END IF;
return new;
end;
$$;


ALTER FUNCTION public.cu_report_visualization_before_update() OWNER TO postgres;

--
-- TOC entry 308 (class 1255 OID 23542)
-- Name: cu_schedule_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_schedule_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_schedule_before_insert() OWNER TO postgres;

--
-- TOC entry 309 (class 1255 OID 23543)
-- Name: cu_schedule_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_schedule_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_schedule_before_update() OWNER TO postgres;

--
-- TOC entry 310 (class 1255 OID 23544)
-- Name: cu_schedule_log_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_schedule_log_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_schedule_log_before_insert() OWNER TO postgres;

--
-- TOC entry 311 (class 1255 OID 23545)
-- Name: cu_schedule_log_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_schedule_log_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_schedule_log_before_update() OWNER TO postgres;

--
-- TOC entry 312 (class 1255 OID 23546)
-- Name: cu_shared_connection_access_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_shared_connection_access_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_shared_connection_access_before_insert() OWNER TO postgres;

--
-- TOC entry 313 (class 1255 OID 23547)
-- Name: cu_shared_visualization_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_shared_visualization_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=now();
return new;
end;
$$;


ALTER FUNCTION public.cu_shared_visualization_before_insert() OWNER TO postgres;

--
-- TOC entry 314 (class 1255 OID 23548)
-- Name: cu_shared_visualization_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_shared_visualization_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (NEW.status ='D') THEN
NEW.status =CONCAT(NEW.status,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.cu_shared_visualization_before_update() OWNER TO postgres;

--
-- TOC entry 315 (class 1255 OID 23549)
-- Name: cu_user_groups_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_user_groups_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.cu_user_groups_before_insert() OWNER TO postgres;

--
-- TOC entry 316 (class 1255 OID 23550)
-- Name: cu_user_groups_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.cu_user_groups_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (NEW.status ='D') THEN
NEW.status =CONCAT(NEW.status,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.cu_user_groups_before_update() OWNER TO postgres;

--
-- TOC entry 317 (class 1255 OID 23551)
-- Name: data_entity_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.data_entity_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (OLD.status !='D' and NEW.status ='D') THEN
NEW.entityname=CONCAT(OLD.entityname,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.data_entity_before_update() OWNER TO postgres;

--
-- TOC entry 330 (class 1255 OID 23552)
-- Name: data_hub_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.data_hub_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.updated_date=NOW();
if (OLD.status !='D' and NEW.status ='D') THEN
NEW.data_hub_name=CONCAT(OLD.data_hub_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.data_hub_before_update() OWNER TO postgres;

--
-- TOC entry 331 (class 1255 OID 23553)
-- Name: data_model_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.data_model_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (OLD.status !='D' and NEW.status ='D') THEN
NEW.name=CONCAT(OLD.name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.data_model_before_update() OWNER TO postgres;

--
-- TOC entry 332 (class 1255 OID 23554)
-- Name: data_model_entity_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.data_model_entity_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (OLD.status !='D' and NEW.status ='D') THEN
NEW.entity_name=CONCAT(OLD.entity_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.data_model_entity_before_update() OWNER TO postgres;

--
-- TOC entry 333 (class 1255 OID 23555)
-- Name: es_temp_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.es_temp_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.es_temp_before_insert() OWNER TO postgres;

--
-- TOC entry 334 (class 1255 OID 23556)
-- Name: es_temp_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.es_temp_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.es_temp_before_update() OWNER TO postgres;

--
-- TOC entry 335 (class 1255 OID 23557)
-- Name: ldap_configurations_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.ldap_configurations_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.ldap_configurations_before_insert() OWNER TO postgres;

--
-- TOC entry 336 (class 1255 OID 23558)
-- Name: ldap_configurations_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.ldap_configurations_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (NEW.status ='D') THEN
NEW.ldap_name =CONCAT(OLD.ldap_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.ldap_configurations_before_update() OWNER TO postgres;

--
-- TOC entry 337 (class 1255 OID 23559)
-- Name: user_account_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.user_account_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.user_account_before_insert() OWNER TO postgres;

--
-- TOC entry 338 (class 1255 OID 23560)
-- Name: user_account_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.user_account_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
IF (NEW.status ='D') THEN
NEW.user_login_name =CONCAT(OLD.user_login_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.user_account_before_update() OWNER TO postgres;

--
-- TOC entry 339 (class 1255 OID 23561)
-- Name: user_role_before_insert(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.user_role_before_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.create_date=NOW();
return new;
end;
$$;


ALTER FUNCTION public.user_role_before_insert() OWNER TO postgres;

--
-- TOC entry 340 (class 1255 OID 23562)
-- Name: user_role_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.user_role_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (NEW.status ='D') THEN
NEW.role_name =CONCAT(OLD.role_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.user_role_before_update() OWNER TO postgres;

--
-- TOC entry 341 (class 1255 OID 23563)
-- Name: workspace_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.workspace_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (OLD.status !='D' and NEW.status ='D') THEN
NEW.workspace_name=CONCAT(OLD.workspace_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.workspace_before_update() OWNER TO postgres;

--
-- TOC entry 342 (class 1255 OID 23564)
-- Name: workspace_entity_before_update(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.workspace_entity_before_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$begin
NEW.update_date=NOW();
if (OLD.status !='D' and NEW.status ='D') THEN
NEW.entity_name=CONCAT(OLD.entity_name,'_', now());
END IF;
return new;
end;
$$;


ALTER FUNCTION public.workspace_entity_before_update() OWNER TO postgres;

SET default_tablespace = '';

--
-- TOC entry 196 (class 1259 OID 23565)
-- Name: api_table; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.api_table (
    id integer NOT NULL,
    path character varying(100),
    endpoint character varying(100)
);


ALTER TABLE public.api_table OWNER TO postgres;

--
-- TOC entry 197 (class 1259 OID 23568)
-- Name: bird_reserved_words_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.bird_reserved_words_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.bird_reserved_words_id OWNER TO postgres;

--
-- TOC entry 198 (class 1259 OID 23570)
-- Name: bird_reserved_words; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bird_reserved_words (
    bird_reserved_words_id integer DEFAULT nextval('public.bird_reserved_words_id'::regclass) NOT NULL,
    reserved_word_name character varying(1000)
);


ALTER TABLE public.bird_reserved_words OWNER TO postgres;

--
-- TOC entry 199 (class 1259 OID 23577)
-- Name: client_license; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.client_license (
    id integer NOT NULL,
    client_id integer,
    license_key text NOT NULL,
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone
);


ALTER TABLE public.client_license OWNER TO postgres;

--
-- TOC entry 200 (class 1259 OID 23583)
-- Name: client_license_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.client_license_audit (
    id integer NOT NULL,
    client_id integer,
    license_key text NOT NULL,
    event_type character varying(10) NOT NULL,
    event_by character varying(50),
    event_date timestamp without time zone NOT NULL
);


ALTER TABLE public.client_license_audit OWNER TO postgres;

--
-- TOC entry 201 (class 1259 OID 23589)
-- Name: client_license_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.client_license_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.client_license_id_seq OWNER TO postgres;

--
-- TOC entry 3790 (class 0 OID 0)
-- Dependencies: 201
-- Name: client_license_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.client_license_id_seq OWNED BY public.client_license.id;


--
-- TOC entry 202 (class 1259 OID 23591)
-- Name: client_partner; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.client_partner (
    client_id integer NOT NULL,
    client_name character varying(255) NOT NULL,
    client_desc character varying(2000),
    partner_id integer,
    is_partner character varying(1) DEFAULT 'N'::character varying NOT NULL,
    client_access_type character varying(100),
    client_access_key character varying(100),
    client_admin_email character varying(100),
    partner_admin_email character varying(100),
    status character varying(1) DEFAULT 'A'::character varying,
    comments character varying(3000),
    create_date timestamp without time zone,
    create_by character varying(100),
    update_date timestamp without time zone,
    update_by character varying(100),
    mail_smtp character varying(100),
    mail_port integer,
    mail_email character varying(100),
    mail_authenticate character varying(100),
    mail_password character varying(100),
    mail_connection_type character varying(100),
    cp_var1 character varying(100),
    cp_var2 character varying(100),
    cp_var3 character varying(100),
    cp_var4 character varying(100),
    cp_var5 character varying(100),
    bird_message character varying(225) DEFAULT 'You have successfully Registered with BIRD. Please click the below link to login'::character varying,
    audit_enabled integer,
    data_age_number integer,
    data_age_duration character varying(50) DEFAULT NULL::character varying,
    singlesignon_config text DEFAULT '{}'::text,
    email_disclaimer_message text DEFAULT ' '::text NOT NULL,
    datahub_db_name text DEFAULT 'defaulthub'::text NOT NULL,
    dataws_db_name text DEFAULT 'defaultws'::text NOT NULL,
    datadm_db_name text DEFAULT 'defaultdm'::text NOT NULL
);


ALTER TABLE public.client_partner OWNER TO postgres;

--
-- TOC entry 203 (class 1259 OID 23606)
-- Name: client_partner_client_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.client_partner_client_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.client_partner_client_id_seq OWNER TO postgres;

--
-- TOC entry 3793 (class 0 OID 0)
-- Dependencies: 203
-- Name: client_partner_client_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.client_partner_client_id_seq OWNED BY public.client_partner.client_id;


--
-- TOC entry 204 (class 1259 OID 23608)
-- Name: cp_feature_access; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cp_feature_access (
    cp_feature_access_id integer NOT NULL,
    feature_id integer,
    group_id integer,
    feature_display_name character varying(100),
    client_id integer,
    role_id integer,
    permissions text,
    user_id integer,
    status character varying(1) DEFAULT 'A'::character varying,
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone
);


ALTER TABLE public.cp_feature_access OWNER TO postgres;

--
-- TOC entry 205 (class 1259 OID 23615)
-- Name: cp_feature_access_cp_feature_access_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cp_feature_access_cp_feature_access_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cp_feature_access_cp_feature_access_id_seq OWNER TO postgres;

--
-- TOC entry 3796 (class 0 OID 0)
-- Dependencies: 205
-- Name: cp_feature_access_cp_feature_access_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cp_feature_access_cp_feature_access_id_seq OWNED BY public.cp_feature_access.cp_feature_access_id;


--
-- TOC entry 206 (class 1259 OID 23617)
-- Name: cp_features; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cp_features (
    feature_id integer NOT NULL,
    feature_name character varying(100),
    feature_display_name character varying(100),
    feature_desc character varying(45),
    parent_feature_id integer,
    feature_order integer,
    status character varying(1) DEFAULT 'A'::character varying,
    client_id integer,
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone
);


ALTER TABLE public.cp_features OWNER TO postgres;

--
-- TOC entry 207 (class 1259 OID 23621)
-- Name: cp_features_feature_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cp_features_feature_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cp_features_feature_id_seq OWNER TO postgres;

--
-- TOC entry 3799 (class 0 OID 0)
-- Dependencies: 207
-- Name: cp_features_feature_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cp_features_feature_id_seq OWNED BY public.cp_features.feature_id;


--
-- TOC entry 208 (class 1259 OID 23623)
-- Name: cp_groups; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cp_groups (
    group_id integer NOT NULL,
    group_name character varying(100),
    group_desc character varying(100),
    status character varying(1) DEFAULT 'A'::character varying,
    client_id integer,
    role_id integer,
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone
);


ALTER TABLE public.cp_groups OWNER TO postgres;

--
-- TOC entry 209 (class 1259 OID 23627)
-- Name: cp_groups_group_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cp_groups_group_id_seq
    START WITH 4
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cp_groups_group_id_seq OWNER TO postgres;

--
-- TOC entry 3802 (class 0 OID 0)
-- Dependencies: 209
-- Name: cp_groups_group_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cp_groups_group_id_seq OWNED BY public.cp_groups.group_id;


--
-- TOC entry 210 (class 1259 OID 23629)
-- Name: cu_alert; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_alert (
    cu_alert_id integer NOT NULL,
    report_visualization_id integer,
    user_id integer,
    client_id integer,
    alert_name character varying(100),
    alert_type character varying(100),
    alert_value text,
    alert_condition text,
    alert_message character varying(300),
    alert_details text,
    status character varying(1) DEFAULT 'A'::character varying,
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone,
    cp_var1 character varying(100),
    cp_var2 character varying(100),
    cp_var3 character varying(100)
);


ALTER TABLE public.cu_alert OWNER TO postgres;

--
-- TOC entry 211 (class 1259 OID 23636)
-- Name: cu_alert_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_alert_audit (
    alert_id integer,
    report_visualization_name character varying(100) DEFAULT NULL::character varying,
    cu_alert_publishinfo_id integer,
    user_id integer,
    client_id integer,
    alert_name character varying(100) DEFAULT NULL::character varying,
    alert_type character varying(100) DEFAULT NULL::character varying,
    alert_value text,
    alert_condition text,
    alert_message character varying(300) DEFAULT NULL::character varying,
    alert_details text,
    published integer DEFAULT 0,
    alert_read integer DEFAULT 0,
    read_time timestamp without time zone,
    published_time timestamp without time zone,
    status character varying(1) DEFAULT 'A'::character varying,
    create_by character varying(100) DEFAULT NULL::character varying,
    create_date timestamp without time zone,
    update_by character varying(100) DEFAULT NULL::character varying,
    update_date timestamp without time zone,
    cp_var1 character varying(100) DEFAULT NULL::character varying,
    cp_var2 character varying(100) DEFAULT NULL::character varying,
    cp_var3 character varying(100) DEFAULT NULL::character varying,
    event_type character varying(10) DEFAULT NULL::character varying,
    event_by character varying(100) DEFAULT NULL::character varying,
    event_date timestamp without time zone
);


ALTER TABLE public.cu_alert_audit OWNER TO postgres;

--
-- TOC entry 212 (class 1259 OID 23656)
-- Name: cu_alert_cu_alert_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_alert_cu_alert_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_alert_cu_alert_id_seq OWNER TO postgres;

--
-- TOC entry 3806 (class 0 OID 0)
-- Dependencies: 212
-- Name: cu_alert_cu_alert_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_alert_cu_alert_id_seq OWNED BY public.cu_alert.cu_alert_id;


--
-- TOC entry 213 (class 1259 OID 23658)
-- Name: cu_alert_publishinfo; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_alert_publishinfo (
    cu_alert_publishinfo_id integer NOT NULL,
    alert_id integer,
    user_id integer,
    client_id integer,
    status character varying(1) DEFAULT 'A'::character varying,
    published integer DEFAULT 0,
    alert_read integer DEFAULT 0,
    published_time timestamp without time zone,
    read_time timestamp without time zone,
    cp_var1 character varying(100),
    cp_var2 character varying(100),
    cp_var3 character varying(100),
    update_date timestamp without time zone,
    update_by character varying(100) DEFAULT NULL::character varying,
    create_by character varying(100) DEFAULT NULL::character varying,
    create_date timestamp without time zone,
    updated integer DEFAULT 0
);


ALTER TABLE public.cu_alert_publishinfo OWNER TO postgres;

--
-- TOC entry 214 (class 1259 OID 23670)
-- Name: cu_alert_publishinfo_cu_alert_publishinfo_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_alert_publishinfo_cu_alert_publishinfo_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_alert_publishinfo_cu_alert_publishinfo_id_seq OWNER TO postgres;

--
-- TOC entry 3809 (class 0 OID 0)
-- Dependencies: 214
-- Name: cu_alert_publishinfo_cu_alert_publishinfo_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_alert_publishinfo_cu_alert_publishinfo_id_seq OWNED BY public.cu_alert_publishinfo.cu_alert_publishinfo_id;


--
-- TOC entry 215 (class 1259 OID 23672)
-- Name: cu_alert_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_alert_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_alert_seq OWNER TO postgres;

--
-- TOC entry 216 (class 1259 OID 23674)
-- Name: cu_connection_access; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_connection_access (
    connection_access_id integer NOT NULL,
    connection_id integer DEFAULT 0 NOT NULL,
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    connection_access_name character varying(100) NOT NULL,
    connection_json text NOT NULL,
    status character varying(1) DEFAULT 'A'::character varying NOT NULL,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    acc_var1 character varying(50),
    acc_var2 character varying(50),
    acc_var3 character varying(50),
    create_by character varying(100),
    update_by character varying(100),
    isfrom_model integer DEFAULT 0
);


ALTER TABLE public.cu_connection_access OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 23683)
-- Name: cu_connection_access_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_connection_access_audit (
    connection_access_id integer NOT NULL,
    connection_id integer DEFAULT 0 NOT NULL,
    client_id integer NOT NULL,
    connection_access_name character varying(100) NOT NULL,
    connection_json text NOT NULL,
    status character varying(1) DEFAULT 'A'::character varying NOT NULL,
    acc_var1 character varying(50),
    acc_var2 character varying(50),
    acc_var3 character varying(50),
    event_type character varying(10) NOT NULL,
    event_by character varying(50),
    event_date timestamp without time zone NOT NULL
);


ALTER TABLE public.cu_connection_access_audit OWNER TO postgres;

--
-- TOC entry 218 (class 1259 OID 23691)
-- Name: cu_connection_access_connection_access_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_connection_access_connection_access_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_connection_access_connection_access_id_seq OWNER TO postgres;

--
-- TOC entry 3814 (class 0 OID 0)
-- Dependencies: 218
-- Name: cu_connection_access_connection_access_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_connection_access_connection_access_id_seq OWNED BY public.cu_connection_access.connection_access_id;


--
-- TOC entry 219 (class 1259 OID 23693)
-- Name: cu_connection_types; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_connection_types (
    type_id integer NOT NULL,
    typename character varying(50),
    typedisplayname character varying(50),
    frequently_used integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.cu_connection_types OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 23697)
-- Name: cu_connection_types_type_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_connection_types_type_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_connection_types_type_id_seq OWNER TO postgres;

--
-- TOC entry 3817 (class 0 OID 0)
-- Dependencies: 220
-- Name: cu_connection_types_type_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_connection_types_type_id_seq OWNED BY public.cu_connection_types.type_id;


--
-- TOC entry 221 (class 1259 OID 23699)
-- Name: cu_connections; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_connections (
    connections_id integer NOT NULL,
    client_id integer,
    connection_name character varying(100),
    connection_display_name character varying(100),
    connection_details_json text,
    connection_type text,
    connection_type_id integer,
    create_date timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.cu_connections OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 23706)
-- Name: cu_connections_connections_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_connections_connections_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_connections_connections_id_seq OWNER TO postgres;

--
-- TOC entry 3820 (class 0 OID 0)
-- Dependencies: 222
-- Name: cu_connections_connections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_connections_connections_id_seq OWNED BY public.cu_connections.connections_id;


--
-- TOC entry 223 (class 1259 OID 23708)
-- Name: cu_dashboard; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_dashboard (
    cu_dashboard_id integer NOT NULL,
    user_id integer,
    client_id integer,
    cu_report_visualization_id integer,
    dashboard_name character varying(100),
    dashboard_desc character varying(100),
    folder_name character varying(100),
    dashboard_details_xml text,
    widget_details_xml text,
    is_default character varying(1) DEFAULT 'N'::character varying,
    status character varying(1) DEFAULT 'A'::character varying,
    is_shared character varying(1) DEFAULT 'N'::character varying,
    shared_to_users text,
    shared_comments text,
    shared_xml text,
    cuud_var1 character varying(45),
    cuud_var2 character varying(45),
    cuud_var3 character varying(45),
    cuud_var4 character varying(45),
    cuud_var5 character varying(45),
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone
);


ALTER TABLE public.cu_dashboard OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 23717)
-- Name: cu_dashboard_cu_dashboard_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_dashboard_cu_dashboard_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_dashboard_cu_dashboard_id_seq OWNER TO postgres;

--
-- TOC entry 3823 (class 0 OID 0)
-- Dependencies: 224
-- Name: cu_dashboard_cu_dashboard_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_dashboard_cu_dashboard_id_seq OWNED BY public.cu_dashboard.cu_dashboard_id;


--
-- TOC entry 225 (class 1259 OID 23719)
-- Name: cu_hash_criteria_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_hash_criteria_id_seq
    START WITH 26
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_hash_criteria_id_seq OWNER TO postgres;

--
-- TOC entry 226 (class 1259 OID 23721)
-- Name: cu_hash_criteria; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_hash_criteria (
    cu_hash_criteria_id integer DEFAULT nextval('public.cu_hash_criteria_id_seq'::regclass) NOT NULL,
    user_id integer,
    client_id integer,
    report_visualization_id integer,
    criteria_title character varying(100),
    hash_criteria_data text,
    create_date timestamp without time zone
);


ALTER TABLE public.cu_hash_criteria OWNER TO postgres;

--
-- TOC entry 227 (class 1259 OID 23728)
-- Name: cu_report_visualization_cu_report_visualization_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_report_visualization_cu_report_visualization_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_report_visualization_cu_report_visualization_id_seq OWNER TO postgres;

--
-- TOC entry 228 (class 1259 OID 23730)
-- Name: cu_report_visualization; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_report_visualization (
    cu_report_visualization_id integer DEFAULT nextval('public.cu_report_visualization_cu_report_visualization_id_seq'::regclass) NOT NULL,
    user_id integer,
    client_id integer,
    report_visualization_name character varying(100),
    report_visualization_desc character varying(4000),
    is_storyboard character varying(1) DEFAULT 'Y'::character varying,
    es_report_type character varying(50),
    es_report_index character varying(200),
    report_title character varying(1000),
    folder_name character varying(45),
    report_columns text,
    user_favorite character varying(1) DEFAULT 'N'::character varying,
    is_shared character varying(1) DEFAULT 'N'::character varying,
    shared_to_users text,
    shared_comments text,
    view_count integer DEFAULT 1,
    curt_var1 character varying(45),
    curt_var2 character varying(45),
    curt_var3 character varying(45),
    curt_var4 character varying(45),
    curt_var5 character varying(45),
    status character varying(1) DEFAULT 'A'::character varying,
    visualization_details_json text,
    visualization_type character varying(45),
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone,
    filter_object text,
    view_only integer,
    meta_data_audit integer DEFAULT 0,
    filter_script text,
    custom_fields_script text,
    drill_through_reports text,
    smart_insights_data text,
    ml_visualization_details_json text,
    ml_model_data text,
    ml_snapshot_data text,
    is_ml_saved integer DEFAULT 0,
    ml_last_run_time timestamp without time zone,
    ml_error_info text,
    renamed_field_json text,
    custom_fields_json text,
    custom_measures_json text,
    data_model_id integer,
    updated_changes_in_report text,
    smart_insight_model_data text,
    smart_insight_visualization_details_json text,
    is_smart_insight_saved integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.cu_report_visualization OWNER TO postgres;

--
-- TOC entry 229 (class 1259 OID 23745)
-- Name: cu_report_visualization_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_report_visualization_audit (
    id integer NOT NULL,
    user_id integer,
    client_id integer,
    dashboard integer DEFAULT 0,
    report_visualization_name text,
    report_visualization_desc character varying(4000),
    is_storyboard character varying(1) DEFAULT 'Y'::character varying,
    es_report_type character varying(50),
    es_report_index character varying(200),
    report_title character varying(1000),
    folder_name character varying(45),
    report_columns text,
    user_favorite character varying(1) DEFAULT 'N'::character varying,
    is_shared character varying(1) DEFAULT 'N'::character varying,
    shared_to_users text,
    shared_to_groups text,
    shared_comments text,
    report_image_location character varying(100),
    view_count integer DEFAULT 1,
    curt_var1 character varying(45),
    curt_var2 character varying(45),
    curt_var3 character varying(45),
    curt_var4 character varying(45),
    curt_var5 character varying(45),
    status character varying(1) DEFAULT 'A'::character varying,
    visualization_details_json text,
    visualization_type character varying(45),
    filter_object text,
    view_only integer,
    event_type character varying(10),
    event_by character varying(100),
    event_date timestamp without time zone,
    updated_changes_in_report text
);


ALTER TABLE public.cu_report_visualization_audit OWNER TO postgres;

--
-- TOC entry 230 (class 1259 OID 23757)
-- Name: cu_schedule; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_schedule (
    cu_schedule_id integer NOT NULL,
    cu_report_visualization_id integer NOT NULL,
    client_id integer,
    user_id integer,
    status character varying(50),
    toemail character varying(500),
    schedule character varying(200),
    job_id character varying(200),
    startdate date,
    enddate date,
    starttime character varying(10),
    endtime character varying(10),
    filters text,
    every character varying(50),
    access_permissions text,
    report_url character varying(400),
    report_name character varying(100),
    create_date timestamp without time zone,
    create_by character varying(50),
    update_date timestamp without time zone,
    update_by character varying(50),
    cus_var1 character varying(50),
    cus_var2 character varying(50),
    cus_var3 character varying(50),
    cus_var4 character varying(50),
    cus_var5 character varying(50),
    fromemail character varying(100)
);


ALTER TABLE public.cu_schedule OWNER TO postgres;

--
-- TOC entry 231 (class 1259 OID 23763)
-- Name: cu_schedule_cu_schedule_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_schedule_cu_schedule_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_schedule_cu_schedule_id_seq OWNER TO postgres;

--
-- TOC entry 3831 (class 0 OID 0)
-- Dependencies: 231
-- Name: cu_schedule_cu_schedule_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_schedule_cu_schedule_id_seq OWNED BY public.cu_schedule.cu_schedule_id;


--
-- TOC entry 232 (class 1259 OID 23765)
-- Name: cu_schedule_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_schedule_log (
    cu_schedule_log_id integer NOT NULL,
    cu_schedule_id integer,
    client_id integer,
    user_id integer,
    status character varying(50),
    schedule_run_status character varying(50),
    job_id character varying(200),
    create_date timestamp without time zone,
    create_by character varying(100),
    update_date timestamp without time zone,
    update_by character varying(100),
    cus_var1 character varying(50),
    cus_var2 character varying(50),
    cus_var3 character varying(50),
    cus_var4 character varying(50),
    cus_var5 character varying(50)
);


ALTER TABLE public.cu_schedule_log OWNER TO postgres;

--
-- TOC entry 233 (class 1259 OID 23771)
-- Name: cu_schedule_log_cu_schedule_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_schedule_log_cu_schedule_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_schedule_log_cu_schedule_log_id_seq OWNER TO postgres;

--
-- TOC entry 3834 (class 0 OID 0)
-- Dependencies: 233
-- Name: cu_schedule_log_cu_schedule_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_schedule_log_cu_schedule_log_id_seq OWNED BY public.cu_schedule_log.cu_schedule_log_id;


--
-- TOC entry 234 (class 1259 OID 23773)
-- Name: cu_shared_connection_access; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_shared_connection_access (
    cu_shared_connection_access_id integer NOT NULL,
    cu_connection_access_id integer,
    user_id integer,
    client_id integer,
    status character varying(1) DEFAULT 'A'::character varying,
    create_by character varying(100),
    create_date timestamp without time zone,
    update_by character varying(100),
    update_date timestamp without time zone
);


ALTER TABLE public.cu_shared_connection_access OWNER TO postgres;

--
-- TOC entry 235 (class 1259 OID 23777)
-- Name: cu_shared_connection_access_cu_shared_connection_access_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_shared_connection_access_cu_shared_connection_access_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_shared_connection_access_cu_shared_connection_access_id_seq OWNER TO postgres;

--
-- TOC entry 3837 (class 0 OID 0)
-- Dependencies: 235
-- Name: cu_shared_connection_access_cu_shared_connection_access_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_shared_connection_access_cu_shared_connection_access_id_seq OWNED BY public.cu_shared_connection_access.cu_shared_connection_access_id;


--
-- TOC entry 236 (class 1259 OID 23779)
-- Name: cu_shared_model_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_shared_model_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_shared_model_id_seq OWNER TO postgres;

--
-- TOC entry 237 (class 1259 OID 23781)
-- Name: cu_shared_model; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_shared_model (
    id integer DEFAULT nextval('public.cu_shared_model_id_seq'::regclass) NOT NULL,
    user_id integer DEFAULT 0 NOT NULL,
    client_id integer DEFAULT 0 NOT NULL,
    sharedtogroup integer DEFAULT 0 NOT NULL,
    model_id integer DEFAULT 0 NOT NULL,
    comments character varying(100) DEFAULT '0'::character varying NOT NULL,
    status character varying(100) DEFAULT '0'::character varying NOT NULL,
    report_columns text,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    create_by character varying(100),
    update_by character varying(100),
    notification_status integer DEFAULT 0,
    filter_object text,
    view_only integer
);


ALTER TABLE public.cu_shared_model OWNER TO postgres;

--
-- TOC entry 238 (class 1259 OID 23795)
-- Name: cu_shared_model_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_shared_model_audit (
    user_id integer,
    client_id integer,
    model_id integer,
    event_type character varying(1000),
    event_date timestamp without time zone,
    update_date timestamp without time zone,
    user_name character varying(1000),
    create_by character varying(1000),
    update_by character varying(1000),
    sharedtogroup integer,
    sharedtogroup_name character varying(1000),
    data_model_name text,
    comments character varying(1000),
    status character varying(100),
    report_columns text,
    filter_object text,
    notification_status integer DEFAULT 0
);


ALTER TABLE public.cu_shared_model_audit OWNER TO postgres;

--
-- TOC entry 239 (class 1259 OID 23802)
-- Name: cu_shared_visualization; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_shared_visualization (
    id integer NOT NULL,
    user_id integer DEFAULT 0 NOT NULL,
    client_id integer DEFAULT 0 NOT NULL,
    sharedtogroup integer DEFAULT 0 NOT NULL,
    report_visualization_id integer DEFAULT 0 NOT NULL,
    comments character varying(100) DEFAULT '0'::character varying NOT NULL,
    status character varying(100) DEFAULT '0'::character varying NOT NULL,
    report_columns text,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    create_by character varying(100),
    update_by character varying(100),
    notification_status integer DEFAULT 0,
    filter_object text,
    view_only integer
);


ALTER TABLE public.cu_shared_visualization OWNER TO postgres;

--
-- TOC entry 240 (class 1259 OID 23815)
-- Name: cu_shared_visualization_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_shared_visualization_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_shared_visualization_id_seq OWNER TO postgres;

--
-- TOC entry 3843 (class 0 OID 0)
-- Dependencies: 240
-- Name: cu_shared_visualization_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_shared_visualization_id_seq OWNED BY public.cu_shared_visualization.id;


--
-- TOC entry 241 (class 1259 OID 23817)
-- Name: cu_storybook_visualization; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_storybook_visualization (
    storybook_id integer NOT NULL,
    user_id integer DEFAULT 0 NOT NULL,
    client_id integer DEFAULT 0 NOT NULL,
    storybook_name character varying(100) NOT NULL,
    status character varying(100) DEFAULT 'A'::character varying NOT NULL,
    storybook_desc text,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    create_by character varying(100),
    update_by character varying(100),
    visualization_details text
);


ALTER TABLE public.cu_storybook_visualization OWNER TO postgres;

--
-- TOC entry 242 (class 1259 OID 23826)
-- Name: cu_storybook_visualization_storybook_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_storybook_visualization_storybook_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_storybook_visualization_storybook_id_seq OWNER TO postgres;

--
-- TOC entry 3846 (class 0 OID 0)
-- Dependencies: 242
-- Name: cu_storybook_visualization_storybook_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_storybook_visualization_storybook_id_seq OWNED BY public.cu_storybook_visualization.storybook_id;


--
-- TOC entry 243 (class 1259 OID 23828)
-- Name: cu_user_groups; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cu_user_groups (
    id integer NOT NULL,
    user_id integer,
    group_id integer,
    client_id integer,
    status character varying(50) DEFAULT 'A'::character varying,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    create_by character varying(100),
    update_by character varying(100)
);


ALTER TABLE public.cu_user_groups OWNER TO postgres;

--
-- TOC entry 244 (class 1259 OID 23832)
-- Name: cu_user_groups_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cu_user_groups_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cu_user_groups_id_seq OWNER TO postgres;

--
-- TOC entry 3849 (class 0 OID 0)
-- Dependencies: 244
-- Name: cu_user_groups_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cu_user_groups_id_seq OWNED BY public.cu_user_groups.id;


--
-- TOC entry 245 (class 1259 OID 23834)
-- Name: data_entity_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_entity_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.data_entity_id OWNER TO postgres;

--
-- TOC entry 246 (class 1259 OID 23836)
-- Name: data_entity; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_entity (
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    entityname character varying NOT NULL,
    connection_access_id integer,
    indexname character varying(1000),
    type_name character varying(1000),
    entity_columns text,
    excel_meta_data character varying(1000),
    excel_files_order character varying(1000),
    actualquery text,
    savedquery text,
    sync_time_taken character varying(1000),
    unique_columns character varying(1000),
    refresh_type character varying(1000),
    last_executed_timestamp timestamp without time zone,
    data_entity_id integer DEFAULT nextval('public.data_entity_id'::regclass) NOT NULL,
    sync_object character varying(1000),
    status character varying(1) DEFAULT 'A'::character varying,
    entity_type character varying(1000),
    isfrom_model integer DEFAULT 0,
    offsetvalue text,
    created_date timestamp without time zone,
    update_date timestamp without time zone,
    sync_status character varying(15) DEFAULT NULL::character varying,
    source_entityname character varying,
    data_sync_details text,
    create_by character varying(1000),
    update_by character varying(1000)
);


ALTER TABLE public.data_entity OWNER TO postgres;

--
-- TOC entry 247 (class 1259 OID 23846)
-- Name: data_hub_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_hub_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.data_hub_id OWNER TO postgres;

--
-- TOC entry 248 (class 1259 OID 23848)
-- Name: data_hub; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_hub (
    data_hub_id integer DEFAULT nextval('public.data_hub_id'::regclass) NOT NULL,
    data_hub_name character varying(100) DEFAULT NULL::character varying,
    client_id integer,
    user_id integer,
    sync_config character varying(1000) DEFAULT NULL::character varying,
    created_date timestamp without time zone,
    updated_date timestamp without time zone,
    status character varying(1) DEFAULT 'A'::character varying,
    data_sync_details text
);


ALTER TABLE public.data_hub OWNER TO postgres;

--
-- TOC entry 249 (class 1259 OID 23858)
-- Name: data_hub_entity_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_hub_entity_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    MAXVALUE 2147483647
    CACHE 1;


ALTER TABLE public.data_hub_entity_id OWNER TO postgres;

--
-- TOC entry 250 (class 1259 OID 23860)
-- Name: data_hub_entity; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_hub_entity (
    data_hub_entity_id integer DEFAULT nextval('public.data_hub_entity_id'::regclass) NOT NULL,
    entity_id integer NOT NULL,
    data_hub_id integer NOT NULL,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    status character varying(1) DEFAULT 'A'::character varying
);


ALTER TABLE public.data_hub_entity OWNER TO postgres;

--
-- TOC entry 251 (class 1259 OID 23865)
-- Name: data_hub_entity_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_hub_entity_audit (
    data_hub_id integer,
    data_entity_id integer,
    user_id integer,
    client_id integer,
    event_type character varying(10),
    event_date timestamp without time zone,
    data_hub_name character varying(100),
    event_update_date timestamp without time zone,
    data_sync_details character varying(100),
    sync_object character varying(1000),
    entity_name character varying(1000),
    connection_access_id integer,
    connection_name character varying(1000),
    indexname character varying(1000),
    type_name character varying(1000),
    refresh_type character varying(1000),
    sync_time_taken character varying(1000),
    status character varying(10),
    unique_columns character varying(1000),
    create_by character varying(1000),
    update_by character varying(1000)
);


ALTER TABLE public.data_hub_entity_audit OWNER TO postgres;

--
-- TOC entry 252 (class 1259 OID 23871)
-- Name: data_model_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_model_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.data_model_id OWNER TO postgres;

--
-- TOC entry 253 (class 1259 OID 23873)
-- Name: data_model; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_model (
    data_model_id integer DEFAULT nextval('public.data_model_id'::regclass) NOT NULL,
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    name text NOT NULL,
    description text,
    create_date timestamp without time zone,
    create_by character varying,
    update_date timestamp without time zone,
    update_by character varying,
    status character varying,
    multi_fact integer DEFAULT 0,
    workspace_id integer,
    index character varying(1000),
    columns text,
    is_shared character varying(1) DEFAULT 'N'::character varying,
    model_active_status integer DEFAULT 1,
    created_view text
);


ALTER TABLE public.data_model OWNER TO postgres;

--
-- TOC entry 254 (class 1259 OID 23883)
-- Name: data_model_entity_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_model_entity_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.data_model_entity_id OWNER TO postgres;

--
-- TOC entry 255 (class 1259 OID 23885)
-- Name: data_model_entity; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_model_entity (
    data_model_entity_id integer DEFAULT nextval('public.data_model_entity_id'::regclass) NOT NULL,
    data_model_id integer NOT NULL,
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    entity_type character varying(1000),
    query text,
    fact integer DEFAULT 0,
    index character varying(1000),
    type character varying(1000),
    columns text,
    create_date timestamp without time zone,
    create_by character varying(1000),
    update_date timestamp without time zone,
    update_by character varying(1000),
    status character varying(1000),
    workspace_entity_id integer,
    connection_access_id integer,
    customfields text,
    filters text,
    entity_name character varying(1000) DEFAULT NULL::character varying,
    customqueryentityid integer DEFAULT 0
);


ALTER TABLE public.data_model_entity OWNER TO postgres;

--
-- TOC entry 256 (class 1259 OID 23895)
-- Name: data_model_entity_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_model_entity_audit (
    data_model_id integer,
    data_model_entity_id integer,
    workspace_entity_id integer,
    user_id integer,
    client_id integer,
    event_type character varying(1000),
    event_date timestamp without time zone,
    event_update_date timestamp without time zone,
    data_model_name text,
    data_model_description text,
    create_by character varying(1000),
    data_model_entity_name character varying(1000),
    workspace_name character varying(1000),
    indexname character varying(1000),
    data_model_status character varying(10),
    data_model_entity_status character varying(10),
    is_model_multifact integer DEFAULT 0,
    update_by character varying(1000)
);


ALTER TABLE public.data_model_entity_audit OWNER TO postgres;

--
-- TOC entry 257 (class 1259 OID 23902)
-- Name: data_model_entity_relation_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_model_entity_relation_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.data_model_entity_relation_id OWNER TO postgres;

--
-- TOC entry 258 (class 1259 OID 23904)
-- Name: data_model_entity_relation; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_model_entity_relation (
    data_model_entity_relation_id integer DEFAULT nextval('public.data_model_entity_relation_id'::regclass) NOT NULL,
    data_model_entity_id integer NOT NULL,
    parent_entity_id integer NOT NULL,
    join_type character varying(1000),
    create_date timestamp without time zone NOT NULL,
    create_by character varying(1000) NOT NULL,
    update_date timestamp without time zone,
    update_by character varying(1000),
    status character varying,
    data_model_id integer,
    primarytable character varying(1000),
    primarycolumn character varying(1000),
    secondarytable character varying(1000),
    secondarycolumn character varying(1000),
    primaryworkspaceentityid integer,
    secondaryworkspaceentityid integer
);


ALTER TABLE public.data_model_entity_relation OWNER TO postgres;

--
-- TOC entry 259 (class 1259 OID 23911)
-- Name: data_model_entity_relation_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_model_entity_relation_audit (
    data_model_entity_id integer,
    data_model_id integer,
    join_type character varying(1000),
    primarytable character varying(1000),
    primarycolumn character varying(1000),
    secondarytable character varying(1000),
    secondarycolumn character varying(1000),
    status character varying(10)
);


ALTER TABLE public.data_model_entity_relation_audit OWNER TO postgres;

--
-- TOC entry 260 (class 1259 OID 23917)
-- Name: data_model_mf_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_model_mf_id
    START WITH 361
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.data_model_mf_id OWNER TO postgres;

--
-- TOC entry 261 (class 1259 OID 23919)
-- Name: data_model_multifact; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_model_multifact (
    data_model_mf_id integer DEFAULT nextval('public.data_model_mf_id'::regclass) NOT NULL,
    data_model_id integer NOT NULL,
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    index_name text NOT NULL,
    create_date timestamp without time zone,
    create_by character varying,
    update_date timestamp without time zone,
    update_by character varying,
    status character varying,
    multi_fact integer DEFAULT 0,
    factname character varying(1000),
    columns text
);


ALTER TABLE public.data_model_multifact OWNER TO postgres;

--
-- TOC entry 262 (class 1259 OID 23927)
-- Name: es_temp_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.es_temp_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.es_temp_id_seq OWNER TO postgres;

--
-- TOC entry 263 (class 1259 OID 23929)
-- Name: es_temp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.es_temp (
    id integer DEFAULT nextval('public.es_temp_id_seq'::regclass) NOT NULL,
    client_id integer NOT NULL,
    user_id integer,
    es_report_index character varying(200),
    es_temp_type character varying(50),
    query text,
    connection_details text,
    excel_meta_data text,
    report_columns text,
    excel_files_order text,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    create_by character varying(100),
    update_by character varying(100),
    is_direct integer DEFAULT 0,
    custom_fields_script text,
    data_model_id integer,
    reporttype character varying(1000)
);


ALTER TABLE public.es_temp OWNER TO postgres;

--
-- TOC entry 264 (class 1259 OID 23937)
-- Name: export_temp_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.export_temp_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.export_temp_seq OWNER TO postgres;

--
-- TOC entry 265 (class 1259 OID 23939)
-- Name: export_temp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.export_temp (
    export_id integer DEFAULT nextval('public.export_temp_seq'::regclass) NOT NULL,
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    is_storyboard character varying(1) DEFAULT NULL::character varying,
    reportname character varying(100) DEFAULT NULL::character varying,
    type_id integer,
    report_query text,
    report_columns text,
    connection_details text,
    filter_object text,
    visualization_details text,
    cc_connections_id integer,
    connection_name character varying(50) DEFAULT NULL::character varying,
    connection_id integer,
    connection_type character varying(50) DEFAULT NULL::character varying,
    connection_display_name character varying(50) DEFAULT NULL::character varying,
    saved_query text,
    typename character varying(50) DEFAULT NULL::character varying,
    es_report_index character varying(50) DEFAULT NULL::character varying,
    es_report_type character varying(50) DEFAULT NULL::character varying,
    report_title character varying(1000) DEFAULT NULL::character varying,
    excel_meta_data text,
    excel_files_order text,
    sync_cron_expression character varying(50) DEFAULT NULL::character varying,
    visualization_type character varying(50) DEFAULT NULL::character varying,
    offsetvalue text,
    filter_script text,
    custom_fields_script text,
    parentreport_id integer DEFAULT 0,
	data_model_id integer  DEFAULT NULL
);


ALTER TABLE public.export_temp OWNER TO postgres;

--
-- TOC entry 266 (class 1259 OID 23958)
-- Name: ldap_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ldap_audit (
    event_type character varying(10) NOT NULL,
    event_by character varying(100) DEFAULT NULL::character varying,
    event_date timestamp without time zone NOT NULL,
    ldap_id integer,
    ldap_name character varying(100) DEFAULT NULL::character varying,
    client_id integer,
    url character varying(100) DEFAULT NULL::character varying,
    bind_user character varying(50) DEFAULT NULL::character varying,
    bind_password character varying(50) DEFAULT NULL::character varying,
    search_base character varying(50) DEFAULT NULL::character varying,
    id_attribute character varying(50) DEFAULT NULL::character varying,
    status character varying(50) DEFAULT NULL::character varying,
    query character varying(100) DEFAULT NULL::character varying
);


ALTER TABLE public.ldap_audit OWNER TO postgres;

--
-- TOC entry 267 (class 1259 OID 23973)
-- Name: ldap_configurations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ldap_configurations (
    ldap_id integer NOT NULL,
    ldap_name character varying(100) DEFAULT '0'::character varying NOT NULL,
    client_id integer NOT NULL,
    url character varying(255) NOT NULL,
    bind_user character varying(100) NOT NULL,
    bind_password character varying(100) NOT NULL,
    search_base character varying(255) NOT NULL,
    query character varying(255),
    id_attribute character varying(100),
    status character varying(1),
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    create_by character varying(100),
    update_by character varying(100)
);


ALTER TABLE public.ldap_configurations OWNER TO postgres;

--
-- TOC entry 268 (class 1259 OID 23980)
-- Name: ldap_configurations_ldap_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ldap_configurations_ldap_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ldap_configurations_ldap_id_seq OWNER TO postgres;

--
-- TOC entry 3874 (class 0 OID 0)
-- Dependencies: 268
-- Name: ldap_configurations_ldap_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ldap_configurations_ldap_id_seq OWNED BY public.ldap_configurations.ldap_id;


--
-- TOC entry 269 (class 1259 OID 23982)
-- Name: log_patterns; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.log_patterns (
    log_id integer NOT NULL,
    client_id integer,
    scopename character varying(50),
    log_type character varying(50),
    log_pattern text,
    created_by integer,
    log_syntax text,
    log_example text,
    update_date date,
    update_by character varying(100),
    create_date date,
    create_by character varying(100)
);


ALTER TABLE public.log_patterns OWNER TO postgres;

--
-- TOC entry 270 (class 1259 OID 23988)
-- Name: log_patterns_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.log_patterns_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.log_patterns_log_id_seq OWNER TO postgres;

--
-- TOC entry 3877 (class 0 OID 0)
-- Dependencies: 270
-- Name: log_patterns_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.log_patterns_log_id_seq OWNED BY public.log_patterns.log_id;


--
-- TOC entry 271 (class 1259 OID 23990)
-- Name: login_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.login_audit (
    ip_address character varying(45) NOT NULL,
    activity character varying(15) NOT NULL,
    client_id integer,
    status character varying(10) NOT NULL,
    user_id character varying(50) NOT NULL,
    event_date timestamp without time zone NOT NULL
);


ALTER TABLE public.login_audit OWNER TO postgres;

--
-- TOC entry 272 (class 1259 OID 23993)
-- Name: mail_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mail_audit (
    id integer NOT NULL,
    mail_from character varying(50) NOT NULL,
    mail_to character varying(50) NOT NULL,
    purpose character varying(20) NOT NULL,
    status character varying(12) NOT NULL,
    client_id integer,
    event_date timestamp without time zone
);


ALTER TABLE public.mail_audit OWNER TO postgres;

--
-- TOC entry 273 (class 1259 OID 23996)
-- Name: mail_audit_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.mail_audit_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.mail_audit_id_seq OWNER TO postgres;

--
-- TOC entry 3881 (class 0 OID 0)
-- Dependencies: 273
-- Name: mail_audit_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.mail_audit_id_seq OWNED BY public.mail_audit.id;


--
-- TOC entry 274 (class 1259 OID 23998)
-- Name: ml_models; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ml_models (
    id integer NOT NULL,
    model_name character varying(100),
    model_type character varying(50),
    input_parameter text,
    output_parameter text,
    create_date timestamp without time zone,
    row_limit integer DEFAULT 100000
);


ALTER TABLE public.ml_models OWNER TO postgres;

--
-- TOC entry 275 (class 1259 OID 24005)
-- Name: ml_temp_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ml_temp_id_seq
    START WITH 33
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ml_temp_id_seq OWNER TO postgres;

--
-- TOC entry 276 (class 1259 OID 24007)
-- Name: ml_temp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ml_temp (
    id integer DEFAULT nextval('public.ml_temp_id_seq'::regclass) NOT NULL,
    client_id integer NOT NULL,
    user_id integer,
    ml_snapshot_data text,
    ml_model_data text,
    ml_error_info text,
    create_date timestamp without time zone
);


ALTER TABLE public.ml_temp OWNER TO postgres;

--
-- TOC entry 277 (class 1259 OID 24014)
-- Name: password_policy_rules_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.password_policy_rules_id_seq
    START WITH 4
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.password_policy_rules_id_seq OWNER TO postgres;

--
-- TOC entry 278 (class 1259 OID 24016)
-- Name: password_policy_rules; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.password_policy_rules (
    id integer DEFAULT nextval('public.password_policy_rules_id_seq'::regclass) NOT NULL,
    client_id integer NOT NULL,
    min_max_length character varying(10),
    min_no_of_lowercase_characters integer,
    min_no_of_uppercase_characters integer,
    min_no_of_digits integer,
    min_no_of_special_characters integer,
    allow_whitespaces integer,
    retention_period integer DEFAULT 6,
    two_factor_authentication integer,
    no_of_login_faild_attempts integer DEFAULT 0
);


ALTER TABLE public.password_policy_rules OWNER TO postgres;

--
-- TOC entry 279 (class 1259 OID 24022)
-- Name: securityquestions_user_registration; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.securityquestions_user_registration (
    question_id integer NOT NULL,
    question text
);


ALTER TABLE public.securityquestions_user_registration OWNER TO postgres;

--
-- TOC entry 280 (class 1259 OID 24028)
-- Name: semantic_names; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.semantic_names (
    id integer NOT NULL,
    cu_schema_id integer,
    actual_name character varying(200),
    display_name character varying(200),
    display_type character varying(200)
);


ALTER TABLE public.semantic_names OWNER TO postgres;

--
-- TOC entry 281 (class 1259 OID 24034)
-- Name: semantic_names_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.semantic_names_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.semantic_names_id_seq OWNER TO postgres;

--
-- TOC entry 3890 (class 0 OID 0)
-- Dependencies: 281
-- Name: semantic_names_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.semantic_names_id_seq OWNED BY public.semantic_names.id;


--
-- TOC entry 295 (class 1259 OID 24587)
-- Name: smartinsight_temp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.smartinsight_temp (
    id integer NOT NULL,
    client_id integer NOT NULL,
    user_id integer,
    smart_insight_model_data text,
    smart_insight_error_info text,
    create_date timestamp without time zone
);


ALTER TABLE public.smartinsight_temp OWNER TO postgres;

--
-- TOC entry 294 (class 1259 OID 24585)
-- Name: smartinsight_temp_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.smartinsight_temp_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.smartinsight_temp_id_seq OWNER TO postgres;

--
-- TOC entry 3893 (class 0 OID 0)
-- Dependencies: 294
-- Name: smartinsight_temp_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.smartinsight_temp_id_seq OWNED BY public.smartinsight_temp.id;


--
-- TOC entry 282 (class 1259 OID 24036)
-- Name: storage_views; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.storage_views (
    columns text,
    viewname text
);


ALTER TABLE public.storage_views OWNER TO postgres;

--
-- TOC entry 283 (class 1259 OID 24042)
-- Name: user_account; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_account (
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    ldap_id integer,
    role_id integer,
    group_id character varying(100),
    user_first_name character varying(100),
    user_last_name character varying(100),
    user_login_name character varying(100),
    business_layer_enabled integer DEFAULT 1 NOT NULL,
    user_password character varying(100),
    user_email character varying(100),
    user_address character varying(1000),
    user_phone_number character varying(100),
    status character varying(1) DEFAULT 'A'::character varying,
    role_name character varying(100),
    reset_password integer DEFAULT 1,
    create_date timestamp without time zone,
    create_by character varying(100),
    update_date timestamp without time zone,
    update_by character varying(100),
    ua_var1 character varying(45),
    ua_var2 character varying(45),
    ua_var3 character varying(45),
    ua_var4 character varying(45),
    ua_var5 character varying(45),
    last_reset_time timestamp without time zone DEFAULT now(),
    user_ip_address character varying(100),
    one_time_password character varying(100),
    last_login_date_time timestamp without time zone,
    login_faild_attempts integer DEFAULT 0,
    forgot_password integer DEFAULT 0 NOT NULL,
    firsttimeresetpassword integer DEFAULT 0 NOT NULL,
    security_questions_answers text,
    resetpw_from_admin integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.user_account OWNER TO postgres;

--
-- TOC entry 284 (class 1259 OID 24056)
-- Name: user_account_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_account_audit (
    event_type character varying(50) NOT NULL,
    event_by character varying(50),
    event_date timestamp without time zone NOT NULL,
    user_or_group character varying(50) NOT NULL,
    user_first_name character varying(100) NOT NULL,
    user_last_name character varying(50),
    user_login_name character varying(100),
    client_id integer,
    status character varying(1),
    group_name character varying(100) DEFAULT NULL::character varying,
    role_name character varying(50) DEFAULT NULL::character varying
);


ALTER TABLE public.user_account_audit OWNER TO postgres;

--
-- TOC entry 285 (class 1259 OID 24064)
-- Name: user_account_user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_account_user_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_account_user_id_seq OWNER TO postgres;

--
-- TOC entry 3898 (class 0 OID 0)
-- Dependencies: 285
-- Name: user_account_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_account_user_id_seq OWNED BY public.user_account.user_id;


--
-- TOC entry 286 (class 1259 OID 24066)
-- Name: user_role; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_role (
    role_id integer NOT NULL,
    role_name character varying(100),
    role_desc character varying(100),
    role_displayname character varying(100),
    parent_role_id integer,
    status character varying(50) DEFAULT 'A'::character varying NOT NULL,
    client_id integer,
    create_date timestamp without time zone,
    update_date timestamp without time zone,
    create_by character varying(100),
    update_by character varying(100)
);


ALTER TABLE public.user_role OWNER TO postgres;

--
-- TOC entry 287 (class 1259 OID 24073)
-- Name: user_role_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_role_audit (
    role_id integer NOT NULL,
    role_name character varying(100) DEFAULT NULL::character varying,
    role_desc character varying(100) DEFAULT NULL::character varying,
    role_display_name character varying(100) DEFAULT NULL::character varying,
    parent_role_id integer,
    status character varying(50) DEFAULT 'A'::character varying NOT NULL,
    client_id integer,
    event_type character varying(10) NOT NULL,
    event_by character varying(100) DEFAULT NULL::character varying,
    event_date timestamp without time zone NOT NULL
);


ALTER TABLE public.user_role_audit OWNER TO postgres;

--
-- TOC entry 288 (class 1259 OID 24081)
-- Name: user_role_role_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_role_role_id_seq
    START WITH 11
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_role_role_id_seq OWNER TO postgres;

--
-- TOC entry 3902 (class 0 OID 0)
-- Dependencies: 288
-- Name: user_role_role_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_role_role_id_seq OWNED BY public.user_role.role_id;


--
-- TOC entry 289 (class 1259 OID 24083)
-- Name: workspace_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.workspace_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.workspace_id OWNER TO postgres;

--
-- TOC entry 290 (class 1259 OID 24085)
-- Name: workspace; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.workspace (
    workspace_id integer DEFAULT nextval('public.workspace_id'::regclass) NOT NULL,
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    workspace_name text NOT NULL,
    workspace_desc text,
    create_date timestamp without time zone,
    create_by character varying(1000),
    update_date timestamp without time zone,
    update_by character varying(1000),
    status character varying(10),
    hub_id integer NOT NULL,
    is_default integer DEFAULT 0,
    sync_config character varying(1000) DEFAULT NULL::character varying,
    data_sync_details text,
    ws_save_status character varying(10) DEFAULT NULL::character varying
);


ALTER TABLE public.workspace OWNER TO postgres;

--
-- TOC entry 291 (class 1259 OID 24095)
-- Name: workspace_entity_id; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.workspace_entity_id
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.workspace_entity_id OWNER TO postgres;

--
-- TOC entry 292 (class 1259 OID 24097)
-- Name: workspace_entity; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.workspace_entity (
    workspace_entity_id integer DEFAULT nextval('public.workspace_entity_id'::regclass) NOT NULL,
    user_id integer NOT NULL,
    client_id integer NOT NULL,
    entity_name character varying(1000),
    type character varying(1000),
    columns text,
    filters text,
    customfields text,
    query text,
    create_by character varying(1000) NOT NULL,
    create_date timestamp without time zone NOT NULL,
    update_by character varying(1000),
    update_date timestamp without time zone,
    status character varying(10),
    connection_access_id integer,
    fact integer DEFAULT 0,
    workspace_id integer NOT NULL,
    indexname character varying(1000) DEFAULT NULL::character varying,
    data_hub_entity_id integer,
    isfrom_model integer DEFAULT 0,
    ws_entity_save_status character varying(10) DEFAULT NULL::character varying,
    error_info text
);


ALTER TABLE public.workspace_entity OWNER TO postgres;

--
-- TOC entry 293 (class 1259 OID 24108)
-- Name: workspace_entity_audit; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.workspace_entity_audit (
    workspace_id integer,
    workspace_entity_id integer,
    user_id integer,
    client_id integer,
    hub_id integer,
    data_hub_entity_id integer,
    event_type character varying(10),
    event_date timestamp without time zone,
    update_date timestamp without time zone,
    update_by character varying(1000),
    connection_name character varying(1000),
    indexname character varying(1000),
    entity_name character varying(1000),
    workspace_name text,
    workspace_desc text,
    status character varying(10),
    filters text,
    customfields text,
    sync_config character varying(1000) DEFAULT NULL::character varying,
    data_sync_details text,
    create_by character varying(1000)
);


ALTER TABLE public.workspace_entity_audit OWNER TO postgres;

--
-- TOC entry 3145 (class 2604 OID 24115)
-- Name: client_license id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.client_license ALTER COLUMN id SET DEFAULT nextval('public.client_license_id_seq'::regclass);


--
-- TOC entry 3155 (class 2604 OID 24116)
-- Name: client_partner client_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.client_partner ALTER COLUMN client_id SET DEFAULT nextval('public.client_partner_client_id_seq'::regclass);


--
-- TOC entry 3157 (class 2604 OID 24117)
-- Name: cp_feature_access cp_feature_access_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_feature_access ALTER COLUMN cp_feature_access_id SET DEFAULT nextval('public.cp_feature_access_cp_feature_access_id_seq'::regclass);


--
-- TOC entry 3159 (class 2604 OID 24118)
-- Name: cp_features feature_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_features ALTER COLUMN feature_id SET DEFAULT nextval('public.cp_features_feature_id_seq'::regclass);


--
-- TOC entry 3161 (class 2604 OID 24119)
-- Name: cp_groups group_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_groups ALTER COLUMN group_id SET DEFAULT nextval('public.cp_groups_group_id_seq'::regclass);


--
-- TOC entry 3163 (class 2604 OID 24120)
-- Name: cu_alert cu_alert_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_alert ALTER COLUMN cu_alert_id SET DEFAULT nextval('public.cu_alert_cu_alert_id_seq'::regclass);


--
-- TOC entry 3184 (class 2604 OID 24121)
-- Name: cu_alert_publishinfo cu_alert_publishinfo_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_alert_publishinfo ALTER COLUMN cu_alert_publishinfo_id SET DEFAULT nextval('public.cu_alert_publishinfo_cu_alert_publishinfo_id_seq'::regclass);


--
-- TOC entry 3188 (class 2604 OID 24122)
-- Name: cu_connection_access connection_access_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_access ALTER COLUMN connection_access_id SET DEFAULT nextval('public.cu_connection_access_connection_access_id_seq'::regclass);


--
-- TOC entry 3192 (class 2604 OID 24123)
-- Name: cu_connection_types type_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_types ALTER COLUMN type_id SET DEFAULT nextval('public.cu_connection_types_type_id_seq'::regclass);


--
-- TOC entry 3194 (class 2604 OID 24124)
-- Name: cu_connections connections_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connections ALTER COLUMN connections_id SET DEFAULT nextval('public.cu_connections_connections_id_seq'::regclass);


--
-- TOC entry 3198 (class 2604 OID 24125)
-- Name: cu_dashboard cu_dashboard_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_dashboard ALTER COLUMN cu_dashboard_id SET DEFAULT nextval('public.cu_dashboard_cu_dashboard_id_seq'::regclass);


--
-- TOC entry 3215 (class 2604 OID 24126)
-- Name: cu_schedule cu_schedule_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_schedule ALTER COLUMN cu_schedule_id SET DEFAULT nextval('public.cu_schedule_cu_schedule_id_seq'::regclass);


--
-- TOC entry 3216 (class 2604 OID 24127)
-- Name: cu_schedule_log cu_schedule_log_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_schedule_log ALTER COLUMN cu_schedule_log_id SET DEFAULT nextval('public.cu_schedule_log_cu_schedule_log_id_seq'::regclass);


--
-- TOC entry 3218 (class 2604 OID 24128)
-- Name: cu_shared_connection_access cu_shared_connection_access_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_connection_access ALTER COLUMN cu_shared_connection_access_id SET DEFAULT nextval('public.cu_shared_connection_access_cu_shared_connection_access_id_seq'::regclass);


--
-- TOC entry 3235 (class 2604 OID 24129)
-- Name: cu_shared_visualization id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_visualization ALTER COLUMN id SET DEFAULT nextval('public.cu_shared_visualization_id_seq'::regclass);


--
-- TOC entry 3239 (class 2604 OID 24130)
-- Name: cu_storybook_visualization storybook_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_storybook_visualization ALTER COLUMN storybook_id SET DEFAULT nextval('public.cu_storybook_visualization_storybook_id_seq'::regclass);


--
-- TOC entry 3241 (class 2604 OID 24131)
-- Name: cu_user_groups id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_user_groups ALTER COLUMN id SET DEFAULT nextval('public.cu_user_groups_id_seq'::regclass);


--
-- TOC entry 3289 (class 2604 OID 24132)
-- Name: ldap_configurations ldap_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ldap_configurations ALTER COLUMN ldap_id SET DEFAULT nextval('public.ldap_configurations_ldap_id_seq'::regclass);


--
-- TOC entry 3290 (class 2604 OID 24133)
-- Name: log_patterns log_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.log_patterns ALTER COLUMN log_id SET DEFAULT nextval('public.log_patterns_log_id_seq'::regclass);


--
-- TOC entry 3291 (class 2604 OID 24134)
-- Name: mail_audit id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mail_audit ALTER COLUMN id SET DEFAULT nextval('public.mail_audit_id_seq'::regclass);


--
-- TOC entry 3297 (class 2604 OID 24135)
-- Name: semantic_names id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.semantic_names ALTER COLUMN id SET DEFAULT nextval('public.semantic_names_id_seq'::regclass);


--
-- TOC entry 3326 (class 2604 OID 24590)
-- Name: smartinsight_temp id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.smartinsight_temp ALTER COLUMN id SET DEFAULT nextval('public.smartinsight_temp_id_seq'::regclass);


--
-- TOC entry 3306 (class 2604 OID 24136)
-- Name: user_account user_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_account ALTER COLUMN user_id SET DEFAULT nextval('public.user_account_user_id_seq'::regclass);


--
-- TOC entry 3310 (class 2604 OID 24137)
-- Name: user_role role_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_role ALTER COLUMN role_id SET DEFAULT nextval('public.user_role_role_id_seq'::regclass);


--
-- TOC entry 3682 (class 0 OID 23565)
-- Dependencies: 196
-- Data for Name: api_table; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3684 (class 0 OID 23570)
-- Dependencies: 198
-- Data for Name: bird_reserved_words; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.bird_reserved_words VALUES (1, 'abstract');
INSERT INTO public.bird_reserved_words VALUES (2, 'boolean');
INSERT INTO public.bird_reserved_words VALUES (3, 'byte');
INSERT INTO public.bird_reserved_words VALUES (4, 'break');
INSERT INTO public.bird_reserved_words VALUES (5, 'case');
INSERT INTO public.bird_reserved_words VALUES (6, 'this');
INSERT INTO public.bird_reserved_words VALUES (7, 'switch');
INSERT INTO public.bird_reserved_words VALUES (8, 'throw');
INSERT INTO public.bird_reserved_words VALUES (9, 'throws');
INSERT INTO public.bird_reserved_words VALUES (10, 'true');
INSERT INTO public.bird_reserved_words VALUES (11, 'false');
INSERT INTO public.bird_reserved_words VALUES (12, 'transient');
INSERT INTO public.bird_reserved_words VALUES (13, 'try');
INSERT INTO public.bird_reserved_words VALUES (14, 'void');
INSERT INTO public.bird_reserved_words VALUES (15, 'while');
INSERT INTO public.bird_reserved_words VALUES (16, 'private');
INSERT INTO public.bird_reserved_words VALUES (17, 'protected');
INSERT INTO public.bird_reserved_words VALUES (18, 'public');
INSERT INTO public.bird_reserved_words VALUES (19, 'return');
INSERT INTO public.bird_reserved_words VALUES (20, 'synchronized');
INSERT INTO public.bird_reserved_words VALUES (21, 'null');
INSERT INTO public.bird_reserved_words VALUES (22, 'extends');
INSERT INTO public.bird_reserved_words VALUES (23, 'index');
INSERT INTO public.bird_reserved_words VALUES (24, 'new');
INSERT INTO public.bird_reserved_words VALUES (25, 'sum');
INSERT INTO public.bird_reserved_words VALUES (26, 'avg');
INSERT INTO public.bird_reserved_words VALUES (27, 'min');
INSERT INTO public.bird_reserved_words VALUES (28, 'max');
INSERT INTO public.bird_reserved_words VALUES (29, 'globalsum');
INSERT INTO public.bird_reserved_words VALUES (30, 'globalmin');
INSERT INTO public.bird_reserved_words VALUES (31, 'globalavg');
INSERT INTO public.bird_reserved_words VALUES (32, 'globaluniqexact');


--
-- TOC entry 3685 (class 0 OID 23577)
-- Dependencies: 199
-- Data for Name: client_license; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3686 (class 0 OID 23583)
-- Dependencies: 200
-- Data for Name: client_license_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3688 (class 0 OID 23591)
-- Dependencies: 202
-- Data for Name: client_partner; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.client_partner VALUES (1, 'Default Company', 'Default Company', NULL, 'Y', 'on-premises', '', 'admin@admin.com', NULL, 'A', NULL, '2014-04-15 19:11:59', NULL, '2018-02-16 11:56:16', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'You have successfully Registered with BIRD. Please click the below link to login', 0, NULL, NULL, '{}', ' ', 'defaulthub', 'defaultws', 'defaultdm');


--
-- TOC entry 3690 (class 0 OID 23608)
-- Dependencies: 204
-- Data for Name: cp_feature_access; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.cp_feature_access VALUES (1, NULL, NULL, NULL, 1, 1, '[{"features":[{"featureDisplayName":"Groups View","allow":true,"parentFeatureId":4,"featureName":"groups_view"},{"featureDisplayName":"Add Group","allow":true,"parentFeatureId":4,"featureName":"add_group"},{"featureDisplayName":"Edit Group","allow":true,"parentFeatureId":4,"featureName":"edit_group"},{"featureDisplayName":"Delete Group","allow":true,"parentFeatureId":4,"featureName":"delete_group"},{"featureDisplayName":"Users View","allow":true,"parentFeatureId":4,"featureName":"users_view"},{"featureDisplayName":"Add User","allow":true,"parentFeatureId":4,"featureName":"add_user"},{"featureDisplayName":"Edit User","allow":true,"parentFeatureId":4,"featureName":"edit_user"},{"featureDisplayName":"Delete User","allow":true,"parentFeatureId":4,"featureName":"delete_user"},{"featureDisplayName":"Scopes View","allow":true,"parentFeatureId":4,"featureName":"scopes_view"},{"featureDisplayName":"Add Scope","allow":true,"parentFeatureId":4,"featureName":"add_scope"},{"featureDisplayName":"Edit Scope","allow":true,"parentFeatureId":4,"featureName":"edit_scope"},{"featureDisplayName":"Delete Scope","allow":true,"parentFeatureId":4,"featureName":"delete_scope"},{"featureDisplayName":"Scope Access View","allow":true,"parentFeatureId":4,"featureName":"scope_access_view"},{"featureDisplayName":"Add Scope Access","allow":true,"parentFeatureId":4,"featureName":"add_scope_access"},{"featureDisplayName":"Edit Scope Access","allow":true,"parentFeatureId":4,"featureName":"eidit_scope_access"},{"featureDisplayName":"Delete Scope Access","allow":true,"parentFeatureId":4,"featureName":"delete_scope_access"},{"featureDisplayName":"Roles View","allow":true,"parentFeatureId":4,"featureName":"roles_view"},{"featureDisplayName":"Add Role","allow":true,"parentFeatureId":4,"featureName":"add_role"},{"featureDisplayName":"Edit Role","allow":true,"parentFeatureId":4,"featureName":"edit_role"},{"featureDisplayName":"Delete Role","allow":true,"parentFeatureId":4,"featureName":"delete_role"},{"featureDisplayName":"Accounts View","allow":true,"parentFeatureId":4,"featureName":"accounts_view"},{"featureDisplayName":"Add Account","allow":true,"parentFeatureId":4,"featureName":"add_account"},{"featureDisplayName":"Edit Account","allow":true,"parentFeatureId":4,"featureName":"edit_account"},{"featureDisplayName":"Delete Account","allow":true,"parentFeatureId":4,"featureName":"delete_account"},{"featureDisplayName":"Other Settings","allow":true,"parentFeatureId":4,"featureName":"settings_view"},{"featureDisplayName":"License Info","allow":true,"parentFeatureId":4,"featureName":"license_info"},{"featureDisplayName":"License Renew","allow":true,"parentFeatureId":4,"featureName":"license_renew"}],"moduleDisplayName":"Administration","moduleName":"admin"}]', NULL, 'A', NULL, '2018-03-14 14:20:02.986741', NULL, '2016-06-27 12:02:57');
INSERT INTO public.cp_feature_access VALUES (2, NULL, NULL, NULL, 1, 4, '[{"features":[{"featureDisplayName":"New Report","allow":true,"parentFeatureId":1,"featureName":"new_report"},{"featureDisplayName":"Auto View","allow":true,"parentFeatureId":1,"featureName":"auto_view"},{"featureDisplayName":"Report Builder","allow":true,"parentFeatureId":1,"featureName":"report_builder"},{"featureDisplayName":"Edit Query","allow":true,"parentFeatureId":1,"featureName":"edit_query"}],"moduleDisplayName":"Create Report","moduleName":"create_report"},{"features":[{"featureDisplayName":"Add New Tile","allow":true,"parentFeatureId":2,"featureName":"viewreport_add_new_tile"},{"featureDisplayName":"Delete New Tile","allow":true,"parentFeatureId":2,"featureName":"viewreport_delete_new_tile"},{"featureDisplayName":"Max Tile","allow":true,"parentFeatureId":2,"featureName":"viewreport_max_tile"},{"featureDisplayName":"Tile Positioning","allow":true,"parentFeatureId":2,"featureName":"viewreport_adjust_tile"},{"featureDisplayName":"Edit Title","allow":true,"parentFeatureId":2,"featureName":"viewreport_edit_title"},{"featureDisplayName":"Query Info","allow":true,"parentFeatureId":2,"featureName":"viewreport_query_info"},{"featureDisplayName":"Save Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_save_report"},{"featureDisplayName":"Clone Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_clone_report"},{"featureDisplayName":"Filters","allow":true,"parentFeatureId":2,"featureName":"viewreport_filters"},{"featureDisplayName":"Share Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_share_report"},{"featureDisplayName":"Favourites","allow":true,"parentFeatureId":2,"featureName":"viewreport_favourites"},{"featureDisplayName":"Drill Down","allow":true,"parentFeatureId":2,"featureName":"viewreport_drill_down"},{"featureDisplayName":"Save Shared Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_save_shared_report"},{"featureDisplayName":"Color Theme","allow":true,"parentFeatureId":2,"featureName":"viewreport_colortheme"},{"featureDisplayName":"Data Behind Chart","allow":true,"parentFeatureId":2,"featureName":"viewreport_data_behind_chart"},{"featureDisplayName":"View Data","allow":true,"parentFeatureId":2,"featureName":"viewreport_view_data"}],"moduleDisplayName":"Story Board","moduleName":"story_board"},{"features":[{"featureDisplayName":"Filters","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_filters"},{"featureDisplayName":"Drill Down","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_drill_down"},{"featureDisplayName":"Query Info","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_query_info"},{"featureDisplayName":"Title Edit","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_title_edit"},{"featureDisplayName":"Save Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_save_report"},{"featureDisplayName":"Clone Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_clone_report"},{"featureDisplayName":"Share Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_share_report"},{"featureDisplayName":"Favourites","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_favourites"},{"featureDisplayName":"Export Option","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_export_option"},{"featureDisplayName":"Attributes settings","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_attributes_settings"},{"featureDisplayName":"Data Table","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_table"},{"featureDisplayName":"Data Table Export Options","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_table_export"},{"featureDisplayName":"Data Table Column Selection","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_table_column_selection"},{"featureDisplayName":"Save Shared Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_save_shared_report"},{"featureDisplayName":"Color Theme","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_colortheme"},{"featureDisplayName":"Data Behind Chart","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_behind_chart"},{"featureDisplayName":"View Data","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_view_data"},{"featureDisplayName":"Trade Lines","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_trade_lines"},{"featureDisplayName":"Rename Column","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_rename_column"}],"moduleDisplayName":"My Report","moduleName":"my_report"},{"features":[{"featureDisplayName":"Groups View","allow":false,"parentFeatureId":4,"featureName":"groups_view"},{"featureDisplayName":"Add Group","allow":false,"parentFeatureId":4,"featureName":"add_group"},{"featureDisplayName":"Edit Group","allow":false,"parentFeatureId":4,"featureName":"edit_group"},{"featureDisplayName":"Delete Group","allow":false,"parentFeatureId":4,"featureName":"delete_group"},{"featureDisplayName":"Users View","allow":false,"parentFeatureId":4,"featureName":"users_view"},{"featureDisplayName":"Add User","allow":false,"parentFeatureId":4,"featureName":"add_user"},{"featureDisplayName":"Edit User","allow":false,"parentFeatureId":4,"featureName":"edit_user"},{"featureDisplayName":"Delete User","allow":false,"parentFeatureId":4,"featureName":"delete_user"},{"featureDisplayName":"Scopes View","allow":false,"parentFeatureId":4,"featureName":"scopes_view"},{"featureDisplayName":"Add Scope","allow":false,"parentFeatureId":4,"featureName":"add_scope"},{"featureDisplayName":"Edit Scope","allow":false,"parentFeatureId":4,"featureName":"edit_scope"},{"featureDisplayName":"Delete Scope","allow":false,"parentFeatureId":4,"featureName":"delete_scope"},{"featureDisplayName":"Scope Access View","allow":false,"parentFeatureId":4,"featureName":"scope_access_view"},{"featureDisplayName":"Add Scope Access","allow":false,"parentFeatureId":4,"featureName":"add_scope_access"},{"featureDisplayName":"Edit Scope Access","allow":false,"parentFeatureId":4,"featureName":"eidit_scope_access"},{"featureDisplayName":"Delete Scope Access","allow":false,"parentFeatureId":4,"featureName":"delete_scope_access"},{"featureDisplayName":"Roles View","allow":false,"parentFeatureId":4,"featureName":"roles_view"},{"featureDisplayName":"Add Role","allow":false,"parentFeatureId":4,"featureName":"add_role"},{"featureDisplayName":"Edit Role","allow":false,"parentFeatureId":4,"featureName":"edit_role"},{"featureDisplayName":"Delete Role","allow":false,"parentFeatureId":4,"featureName":"delete_role"},{"featureDisplayName":"Accounts View","allow":false,"parentFeatureId":4,"featureName":"accounts_view"},{"featureDisplayName":"Add Account","allow":false,"parentFeatureId":4,"featureName":"add_account"},{"featureDisplayName":"Edit Account","allow":false,"parentFeatureId":4,"featureName":"edit_account"},{"featureDisplayName":"Delete Account","allow":false,"parentFeatureId":4,"featureName":"delete_account"},{"featureDisplayName":"Other Settings","allow":false,"parentFeatureId":4,"featureName":"settings_view"},{"featureDisplayName":"License Info","allow":false,"parentFeatureId":4,"featureName":"license_info"},{"featureDisplayName":"License Renew","allow":false,"parentFeatureId":4,"featureName":"license_renew"}],"moduleDisplayName":"Administration","moduleName":"admin"}]', NULL, 'A', NULL, '2018-03-14 14:20:02.986741', NULL, '2016-06-27 12:02:58');
INSERT INTO public.cp_feature_access VALUES (3, NULL, NULL, NULL, 1, 6, '[{"features":[{"featureDisplayName":"Add New Tile","allow":true,"parentFeatureId":2,"featureName":"viewreport_add_new_tile"},{"featureDisplayName":"Delete New Tile","allow":true,"parentFeatureId":2,"featureName":"viewreport_delete_new_tile"},{"featureDisplayName":"Max Tile","allow":true,"parentFeatureId":2,"featureName":"viewreport_max_tile"},{"featureDisplayName":"Tile Positioning","allow":true,"parentFeatureId":2,"featureName":"viewreport_adjust_tile"},{"featureDisplayName":"Edit Title","allow":true,"parentFeatureId":2,"featureName":"viewreport_edit_title"},{"featureDisplayName":"Query Info","allow":true,"parentFeatureId":2,"featureName":"viewreport_query_info"},{"featureDisplayName":"Save Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_save_report"},{"featureDisplayName":"Clone Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_clone_report"},{"featureDisplayName":"Filters","allow":true,"parentFeatureId":2,"featureName":"viewreport_filters"},{"featureDisplayName":"Share Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_share_report"},{"featureDisplayName":"Favourites","allow":true,"parentFeatureId":2,"featureName":"viewreport_favourites"},{"featureDisplayName":"Drill Down","allow":true,"parentFeatureId":2,"featureName":"viewreport_drill_down"},{"featureDisplayName":"Save Shared Report","allow":true,"parentFeatureId":2,"featureName":"viewreport_save_shared_report"},{"featureDisplayName":"Color Theme","allow":true,"parentFeatureId":2,"featureName":"viewreport_colortheme"},{"featureDisplayName":"Data Behind Chart","allow":true,"parentFeatureId":2,"featureName":"viewreport_data_behind_chart"},{"featureDisplayName":"View Data","allow":true,"parentFeatureId":2,"featureName":"viewreport_view_data"}],"moduleDisplayName":"Story Board","moduleName":"story_board"},{"features":[{"featureDisplayName":"Filters","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_filters"},{"featureDisplayName":"Drill Down","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_drill_down"},{"featureDisplayName":"Query Info","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_query_info"},{"featureDisplayName":"Title Edit","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_title_edit"},{"featureDisplayName":"Save Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_save_report"},{"featureDisplayName":"Clone Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_clone_report"},{"featureDisplayName":"Share Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_share_report"},{"featureDisplayName":"Favourites","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_favourites"},{"featureDisplayName":"Export Option","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_export_option"},{"featureDisplayName":"Attributes settings","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_attributes_settings"},{"featureDisplayName":"Data Table","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_table"},{"featureDisplayName":"Data Table Export Options","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_table_export"},{"featureDisplayName":"Data Table Column Selection","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_table_column_selection"},{"featureDisplayName":"Save Shared Report","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_save_shared_report"},{"featureDisplayName":"Color Theme","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_colortheme"},{"featureDisplayName":"Data Behind Chart","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_data_behind_chart"},{"featureDisplayName":"View Data","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_view_data"},{"featureDisplayName":"Trade Lines","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_trade_lines"},{"featureDisplayName":"Rename Column","allow":true,"parentFeatureId":3,"featureName":"viewreport_maximize_rename_column"}],"moduleDisplayName":"My Report","moduleName":"my_report"}]', NULL, 'A', NULL, '2018-03-14 14:20:02.986741', NULL, '2016-06-27 12:02:58');
INSERT INTO public.cp_feature_access VALUES (4, NULL, NULL, NULL, 1, 7, '[{"features":[{"featureDisplayName":"Accounts View","allow":true,"parentFeatureId":4,"featureName":"accounts_view"},{"featureDisplayName":"Add Account","allow":true,"parentFeatureId":4,"featureName":"add_account"},{"featureDisplayName":"Edit Account","allow":true,"parentFeatureId":4,"featureName":"edit_account"},{"featureDisplayName":"Delete Account","allow":true,"parentFeatureId":4,"featureName":"delete_account"},{"featureDisplayName":"Other Settings","allow":true,"parentFeatureId":4,"featureName":"settings_view"},{"featureDisplayName":"License Info","allow":true,"parentFeatureId":4,"featureName":"license_info"},{"featureDisplayName":"License Renew","allow":true,"parentFeatureId":4,"featureName":"license_renew"}],"moduleDisplayName":"Administration","moduleName":"admin"}]', NULL, 'A', NULL, '2018-03-14 14:20:02.986741', NULL, '2016-06-27 12:02:57');
INSERT INTO public.cp_feature_access VALUES (5, NULL, NULL, NULL, 1, 9, '[{"features":[{"featureDisplayName":"Accounts View","allow":true,"parentFeatureId":4,"featureName":"accounts_view"},{"featureDisplayName":"Add Account","allow":true,"parentFeatureId":4,"featureName":"add_account"},{"featureDisplayName":"Edit Account","allow":true,"parentFeatureId":4,"featureName":"edit_account"},{"featureDisplayName":"Delete Account","allow":true,"parentFeatureId":4,"featureName":"delete_account"},{"featureDisplayName":"Other Settings","allow":true,"parentFeatureId":4,"featureName":"settings_view"},{"featureDisplayName":"License Info","allow":true,"parentFeatureId":4,"featureName":"license_info"},{"featureDisplayName":"License Renew","allow":true,"parentFeatureId":4,"featureName":"license_renew"}],"moduleDisplayName":"Administration","moduleName":"admin"}]', NULL, 'A', NULL, '2018-03-14 14:20:02.986741', NULL, '2016-06-27 12:02:57');
INSERT INTO public.cp_feature_access VALUES (6, NULL, NULL, NULL, 1, 10, '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', NULL, 'A', NULL, '2019-05-13 17:26:37.601365', NULL, NULL);


--
-- TOC entry 3692 (class 0 OID 23617)
-- Dependencies: 206
-- Data for Name: cp_features; Type: TABLE DATA; Schema: public; Owner: postgres

INSERT INTO public.cp_features VALUES (1, 'create_report', 'Create Report', NULL, 0, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (2, 'story_board', 'Story Board', NULL, 0, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (3, 'my_report', 'My Report', NULL, 0, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (4, 'admin', 'Administration', NULL, 0, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (5, 'new_report', 'New Report', NULL,1, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (6, 'viewreport_add_new_tile', 'Add New Tile', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (7, 'viewreport_delete_new_tile', 'Delete New Tile', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (8, 'viewreport_max_tile', 'Max Tile', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (9, 'viewreport_adjust_tile', 'Tile Positioning', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (10, 'viewreport_edit_title', 'Edit Title', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (11, 'viewreport_query_info', 'Query Info', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (12, 'viewreport_save_report', 'Save Report', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (13, 'viewreport_clone_report', 'Clone Report', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (14, 'viewreport_filters', 'Filters', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (15, 'viewreport_share_report', 'Share Report', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (16, 'viewreport_embed_report_and_email', 'Embed Report and Email', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (17, 'viewreport_favourites', 'Favourites', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (18, 'viewreport_drill_down', 'Drill Down', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (19, 'viewreport_save_shared_report', 'Save Shared Report', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (20, 'viewreport_maximize_filters', 'Filters', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (21, 'viewreport_maximize_drill_down', 'Drill Down', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (22, 'viewreport_maximize_query_info', 'Query Info', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (23, 'viewreport_maximize_title_edit', 'Title Edit', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (24, 'viewreport_maximize_save_report', 'Save Report', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (25, 'viewreport_maximize_clone_report', 'Clone Report', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (26, 'viewreport_maximize_share_report', 'Share Report', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (27, 'viewreport_maximize_favourites', 'Favourites', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (28, 'viewreport_maximize_embed_report_email', 'Embed Report and Email', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (29, 'viewreport_maximize_schedule_reports', 'Schedule Report', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (30, 'viewreport_maximize_export_option', 'Export Option', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (31, 'viewreport_maximize_attributes_settings', 'Attribute Settings', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (32, 'viewreport_maximize_data_table', 'Data Table', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (33, 'viewreport_maximize_data_table_export', 'Data Table Export Options', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (34, 'viewreport_maximize_data_table_column_selection', 'Data Table Column Selection', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (35, 'viewreport_maximize_save_shared_report', 'Save Shared Report', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (36, 'groups_view', 'Groups View', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (37, 'add_group', 'Add Group', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (38, 'edit_group', 'Edit Group', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (39, 'delete_group', 'Delete Group', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (40, 'users_view', 'Users View', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (41, 'add_user', 'Add User', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (42, 'edit_user', 'Edit User', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (43, 'delete_user', 'Delete User', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (44, 'roles_view', 'Roles View', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (45, 'add_role', 'Add Role', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (46, 'edit_role', 'Edit Role', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (47, 'delete_role', 'Delete Role', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (48, 'accounts_view', 'Accounts View (Multi-Tenancy)', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (49, 'add_account', 'Add Account (Multi-Tenancy)', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (50, 'edit_account', 'Edit Account (Multi-Tenancy)', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (51, 'delete_account', 'Delete Account (Multi-Tenancy)', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (52, 'settings_view', 'Other Settings', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (53, 'add_ldap', 'Add LDAP', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (54, 'edit_ldap', 'Edit LDAP', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (55, 'delete_ldap', 'Delete LDAP', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (56, 'view_ldap_users', 'View LDAP Users', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (57, 'viewreport_maximize_colortheme', 'Color Theme', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (58, 'viewreport_colortheme', 'Color Theme', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (59, 'viewreport_maximize_pivot', 'Pivot Table', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, '2015-11-24 15:57:39');
INSERT INTO public.cp_features VALUES (60, 'schedule_view', 'Schedule View', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (61, 'edit_schedule', 'Edit Schedule', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (62, 'delete_schedule', 'Delete Schedule', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (63, 'history_schedule', 'Schedule History', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (64, 'viewreport_maximize_pivot_chart', 'Pivot Chart', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, '2015-11-25 13:15:28');
INSERT INTO public.cp_features VALUES (65, 'viewreport_embed_attachment', 'Embed Report Attachment', NULL, 2, NULL, 'A', NULL, NULL, '2016-02-18 12:45:33', NULL, '2016-02-18 12:47:10');
INSERT INTO public.cp_features VALUES (66, 'viewreport_maximize_embed_report_attachment', 'Email Report Attachment', NULL, 3, NULL, 'A', NULL, NULL, '2016-02-18 12:46:55', NULL, NULL);
INSERT INTO public.cp_features VALUES (67, 'viewreport_schedule_reports', 'Schedule Report', NULL, 2, NULL, 'A', NULL, NULL, '2016-02-18 12:48:50', NULL, NULL);
INSERT INTO public.cp_features VALUES (68, 'license_info', 'License Info', NULL, 4, NULL, 'A', NULL, NULL, '2016-02-22 11:52:53', NULL, NULL);
INSERT INTO public.cp_features VALUES (69, 'license_renew', 'License Renew', NULL, 4, NULL, 'A', NULL, NULL, '2016-02-22 11:53:11', NULL, '2016-02-22 11:53:24');
INSERT INTO public.cp_features VALUES (70, 'viewreport_maximize_data_behind_chart', 'Data Behind Chart', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (71, 'viewreport_maximize_data_behind_pivot', 'Data Behind Pivot', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (72, 'viewreport_maximize_view_data', 'View Data', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (73, 'viewreport_maximize_trade_lines', 'Trend Lines', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (74, 'viewreport_view_data', 'View Data', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (75, 'viewreport_data_behind_chart', 'Data Behind Chart', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (76, 'viewreport_maximize_rename_column', 'Rename column', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (77, 'viewreport_maximize_summary_table', 'Summary Table', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (78, 'viewreport_maximize_summary_table_export', 'Summary Table Export', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (80, 'viewreport_maximize_custom_field', 'Custom Field', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (81, 'viewreport_maximize_custom_hierarchy', 'Custom Hierarchy', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (82, 'reset_password', 'Reset Password', NULL, 4, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (83, 'viewreport_maximize_custom_range', 'Custom Range', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (84, 'viewreport_maximize_custom_measure', 'Custom Measure', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (85, 'forcesync', 'Force Sync', NULL, 4, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (86, 'viewreport_clone_tile', 'Clone Tile', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (87, 'viewreport_maximize_alerts', 'Alerts and Notifications', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (88, 'viewreport_maximize_alerts_email', 'Alert Email Notification', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (89, 'viewreport_nlp_search', 'Intelligent Search', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (90, 'viewreport_maximize_nlp_search', 'Intelligent Search', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (91, 'viewreport_embeded_drilldown', 'Embedded Report Drilldown', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (92, 'viewreport_embeded_filters', 'Embedded Report Filters', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (93, 'viewreport_maximize_embeded_drilldown', 'Embedded Report Drilldown', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (94, 'viewreport_maximize_embeded_filters', 'Embedded Report Filters', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (95, 'reportmanagement_reports', 'Reports View', NULL, 4, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (96, 'reportmanagement_connections', 'Connections View', NULL, 4, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (97, 'reportmanagement_share_report', 'Share Report', NULL, 4, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (98, 'reportmanagement_assign_connection', 'Assign Connection', NULL, 4, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (99, 'add_connection_view', 'Add Connection', NULL, 1, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (100, 'download_myreport', 'Download MyReport', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (101, 'download_storyboard', 'Download Storyboard', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (102, 'live_storyboard', 'Live Storyboard', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (103, 'live_report', 'Live Report', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (104, 'admin_view_ldap', 'LDAP View', NULL, 4, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (105, 'admin_reportmanagement_alerts', 'Alerts View', NULL, 4, NULL, 'A', NULL, NULL, '2017-11-26 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (110, 'auditing', 'Auditing', NULL, 4, NULL, 'A', NULL, NULL, '2018-02-07 18:17:02', NULL, NULL);
INSERT INTO public.cp_features VALUES (111, 'viewreport_export_report', 'Export Storyboard', NULL, 2, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES (112, 'viewreport_maximize_export_report', 'Export Report', NULL, 3, NULL, 'A', NULL, NULL, '2015-12-05 15:00:04', NULL, NULL);
INSERT INTO public.cp_features VALUES(113, 'import_template_view', 'Import Template', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (114, 'audit_meta_data', 'Audit Meta Data(StoryBoard)', NULL, 2, NULL, 'A', NULL, NULL, '2018-10-23 18:02:42.171494', NULL, NULL);
INSERT INTO public.cp_features VALUES (115, 'audit_meta_data_maximize', 'Audit Meta Data(Report)', NULL, 3, NULL, 'A', NULL, NULL, '2018-10-23 18:02:42.171494', NULL, NULL);
INSERT INTO public.cp_features VALUES (116, 'newreport_custom_filters_view', 'New Report Custom Filters ', NULL, 1, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (117, 'newreport_custom_fields_view', 'New Report Custom Fields ', NULL, 1, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (118, 'viewreport_drill_through', 'Drill Through ', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (119, 'viewreport_maximize_drill_through', 'Drill Through ', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (120, 'viewreport_drilldown_settings', 'Drilldown Settings ', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (121, 'viewreport_maximize_drilldown_settings', 'Drilldown Settings ', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (127, 'viewreport_alerts', 'Alerts and Notifications', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (129, 'viewreport_custom_script', 'Custom Script ', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (130, 'viewreport_quick_information', 'Quick Information', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (131, 'viewreport_maximize_quick_information', 'Quick Information', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (79, 'viewreport_maximize_prediction', 'Prediction', NULL, 3, NULL, 'D', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (132, 'data_hub', 'Data Hub', NULL, 0, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (133, 'new_data_hub', 'New Data Hub', NULL, 132, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (134, 'work_space', 'Business Workspace', NULL, 0, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (135, 'new_work_space', 'New Workspace', NULL, 134, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (136, 'edit_workspace', 'Edit Workspace', NULL, 134, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (137, 'delete_workspace', 'Delete Workspce', NULL, 134, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (138, 'workspce_model', 'New Business Model', NULL, 134, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (140, 'business_model', 'Business Model', NULL, 0, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (141, 'new_model', 'New Business Model', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (142, 'edit_model', 'Edit Business Model', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (143, 'delete_model', 'Delete Business Model', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (144, 'model_governance', 'Model Governance', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (145, 'create_options_model', 'Create Options', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (146, 'create_model_storyboard', 'Story Board', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (147, 'create_model_report', 'Create Report', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (148, 'create_model_smartinsights', 'Smart Insights', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (149, 'create_model_mlreport', 'Create ML Report', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (150, 'viewreport_pivot_chart', 'Pivot Chart', NULL, 2, NULL, 'A', NULL, NULL, '2020-01-02 11:18:09.83914', NULL, '2020-01-02 11:18:09.83914');
INSERT INTO public.cp_features VALUES (151, 'viewreport_pivot', 'Pivot Table', NULL, 2, NULL, 'A', NULL, NULL, '2020-01-02 11:18:09.83914', NULL, '2020-01-02 11:18:09.83914');
INSERT INTO public.cp_features VALUES (152, 'viewreport_summary_table', 'Summary Table', NULL, 2, NULL, 'A', NULL, NULL, '2020-01-02 11:18:09.83914', NULL, '2020-01-02 11:18:09.83914');
INSERT INTO public.cp_features VALUES (154, 'viewreport_rename_column', 'Rename Column', NULL, 2, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (155, 'viewreport_embeded_view_data', 'Embedded Report View Data', NULL, 2, NULL, 'A', NULL, NULL, '2020-01-08 17:50:27.090841', NULL, '2020-01-08 17:50:27.090841');
INSERT INTO public.cp_features VALUES (156, 'viewreport_maximize_embeded_view_data', 'Embedded Report View Data', NULL, 3, NULL, 'A', NULL, NULL, NULL, NULL, NULL);
INSERT INTO public.cp_features VALUES (157, 'advance_search', 'Advance Search', NULL, 140, NULL, 'A', NULL, NULL, NULL, NULL, NULL);

--
-- TOC entry 3694 (class 0 OID 23623)
-- Dependencies: 208
-- Data for Name: cp_groups; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.cp_groups VALUES (1, 'default', 'This is a default Group', 'A', 1, 6, NULL, '2018-03-14 17:02:12.846025', NULL, '2016-01-09 09:26:41');


--
-- TOC entry 3696 (class 0 OID 23629)
-- Dependencies: 210
-- Data for Name: cu_alert; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3697 (class 0 OID 23636)
-- Dependencies: 211
-- Data for Name: cu_alert_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3699 (class 0 OID 23658)
-- Dependencies: 213
-- Data for Name: cu_alert_publishinfo; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3702 (class 0 OID 23674)
-- Dependencies: 216
-- Data for Name: cu_connection_access; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3703 (class 0 OID 23683)
-- Dependencies: 217
-- Data for Name: cu_connection_access_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3705 (class 0 OID 23693)
-- Dependencies: 219
-- Data for Name: cu_connection_types; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.cu_connection_types VALUES (1, 'database', 'To DataBase', 1);
INSERT INTO public.cu_connection_types VALUES (2, 'flat files', 'To Flat File', 1);
INSERT INTO public.cu_connection_types VALUES (3, 'big data', 'To Big Data', 0);
INSERT INTO public.cu_connection_types VALUES (4, 'service', 'To Service', 0);
INSERT INTO public.cu_connection_types VALUES (5, 'others', 'To Others', 0);
INSERT INTO public.cu_connection_types VALUES (6, 'streaming', 'To Streaming', 0);


--
-- TOC entry 3707 (class 0 OID 23699)
-- Dependencies: 221
-- Data for Name: cu_connections; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.cu_connections VALUES (2, 1, 'CSV', 'CSV', '[{"name":"Upload File","type":"browse","allow":".csv","required":true,"uploadsize":30}]', 'F', 2, '2017-09-15 16:33:02');
INSERT INTO public.cu_connections VALUES (3, 1, 'MySQL', 'MySQL', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TSVC', 1, '2018-01-03 12:56:33');
INSERT INTO public.cu_connections VALUES (4, 1, 'ElasticSearch', 'ElasticSearch', '[{"name":"Connection Name", "type":"text","id":"connectionname", "displaytype":"text", "required":true ,"pattern":true},{"name":"IP Address", "id":"url","type":"text", "displaytype":"text", "required":true }, {"name":"Port","id":"port","type":"text", "displaytype":"number","required":true },{"name":"Cluster Name", "id":"clustername"  , "type":"text", "displaytype":"text","required":true },{"name":"User Name", "id":"username"  , "type":"text", "displaytype":"text", "required":false},{"name":"Password", "id":"password"  , "type":"password", "displaytype":"password", "required":false } ]', 'T', 3, '2018-01-03 12:56:52');
INSERT INTO public.cu_connections VALUES (5, 1, 'SQLServer', 'SQLServer', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TSVC', 1, '2018-01-03 12:57:11');
INSERT INTO public.cu_connections VALUES (6, 1, 'Mongodb', 'MongoDB', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":false},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":false}]', 'T', 3, '2018-01-03 12:57:29');
INSERT INTO public.cu_connections VALUES (9, 1, 'WebDataConnector', 'Web Data Connector', '[{}]', 'WebDataConnector', 4, '2017-09-15 16:35:56');
INSERT INTO public.cu_connections VALUES (10, 1, 'HIVE', 'HIVE', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":false},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":false}]', 'TC', 3, '2018-01-03 12:57:46');
INSERT INTO public.cu_connections VALUES (11, 1, 'RabbitMQ', 'RabbitMQ', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"TCP Port","id":"tcpport","type":"text","displaytype":"number","required":true},{"name":"HTTP Port","id":"httpport","type":"text","displaytype":"number","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":false},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":false},{"name":"Virtualhost","id":"virtualhost","type":"text","displaytype":"text","required":false}]', 'T', 6, '2017-09-15 16:36:45');
INSERT INTO public.cu_connections VALUES (12, 1, 'Oracle', 'Oracle', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TVC', 1, '2018-01-03 12:58:25');
INSERT INTO public.cu_connections VALUES (13, 1, 'Cassandra', 'Cassandra', '[{"name": "Connection Name","type": "text","id": "connectionname","displaytype": "text","required": true,"pattern": true }, {"name": "IP Address","id": "url","type": "text","displaytype": "text","required": true }, {"name": "Port","id": "port","type": "text","displaytype": "number","required": true }, {"name": "DB Name","id": "dbname","type": "text","displaytype": "text","required": true }, {"name": "User Name","id": "dbuser","type": "text","displaytype": "text","required": false }, {"name": "Password","id": "dbPassword","type": "text","displaytype": "password","required": false }]', 'TC', 3, '2018-01-03 12:58:43');
INSERT INTO public.cu_connections VALUES (14, 1, 'Amazon', 'Redshift', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TCV', 3, '2018-01-03 12:59:01');
INSERT INTO public.cu_connections VALUES (15, 1, 'Postgres', 'PostgreSQL', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TVC', 1, '2018-01-03 12:59:15');
INSERT INTO public.cu_connections VALUES (16, 1, 'json', 'JSON', '[{"name":"Upload File","type":"browse","allow":".json","required":true,"uploadsize":30}]', 'JSON', 2, '2017-09-27 12:21:14');
INSERT INTO public.cu_connections VALUES (17, 1, 'EventHub', 'Event Hub', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Consumer Group","id":"consumergroupname","type":"text","displaytype":"text","required":true},{"name":"Name Space","id":"namespacename","type":"text","displaytype":"text","required":true},{"name":"Event Hub Name","id":"eventhubname","type":"text","displaytype":"text","required":true},{"name":"Key Name","id":"keyname","type":"text","displaytype":"text","required":true},{"name":"Primary Key Value","id":"keyvalue","type":"text","displaytype":"password","required":true},{"name": "Number Of Partitions ","id": "numberofpartions","type": "number","displaytype": "number","required": true}]', 'EventHub', 6, '2017-10-13 11:11:02');
INSERT INTO public.cu_connections VALUES (20, 1, 'kafka', 'Kafka', '[]', 'T', 6, '2017-10-16 20:47:06');
INSERT INTO public.cu_connections VALUES (21, 1, 'WebSocket', 'WebSocket', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"URL","id":"url","type":"text","displaytype":"text","required":true}]', 'T', 6, '2017-12-07 11:43:14');
INSERT INTO public.cu_connections VALUES (22, 1, 'log', 'LOG', '[{"name":"Upload File","uploadtype":"browse","allow":".log","required":true,"uploadsize":30}]', 'LOG', 2, '2017-12-14 10:31:28');
INSERT INTO public.cu_connections VALUES (23, 1, 'KUDU', 'Apache Kudu', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"IP Address","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":false},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":false}]', 'TC', 3, '2018-05-28 12:42:37.97375');
INSERT INTO public.cu_connections VALUES (1, 1, 'XLS', 'Excel', '[{"name":"Upload File","type":"browse","allow":".xls,.xlsx","required":true,"uploadsize":10}]', 'FR', 2, '2017-09-15 16:32:25');
INSERT INTO public.cu_connections VALUES (25, 1, 'Pdf', 'PDF', '[{"name":"Upload File","type":"browse","allow":".pdf","required":true,"uploadsize":30}]', 'pdf', 2, '2017-09-15 16:33:02');
INSERT INTO public.cu_connections VALUES (27, 1, 'Neo4J', 'Neo4J', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (29, 1, 'Snowflake', 'Snowflake', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TSVC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (30, 1, 'Zendesk', 'Zendesk', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"URL","id":"url","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'T', 4, '2020-04-30 12:56:33');
INSERT INTO public.cu_connections VALUES (32, 1, 'HSQLDB', 'HSQLDB', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (33, 1, 'TeraData', 'TeraData', '[{"name": "Connection Name","type": "text","id": "connectionname","displaytype": "text","required": true,"pattern": true},{"name": "IP Address","id": "url","type": "text","displaytype": "text","required": true},{"name": "Port","id": "port","type": "text","displaytype": "number","required": true},{"name": "Schema Name","id": "schemaname","type": "text","displaytype": "text","required": true},{"name": "DB Name","id": "dbname","type": "text","displaytype": "text","required": true},{"name": "User Name","id": "dbuser","type": "text","displaytype": "text","required": false},{"name": "Password","id": "dbPassword","type": "text","displaytype": "password","required": false} ]', 'Tc', 3, '2019-06-04 10:25:39.863641');
INSERT INTO public.cu_connections VALUES (35, 1, 'Hana', 'Hana', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (31, 1, 'Presto', 'Presto', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"Catalog Name","id":"catalogname","type":"text","displaytype":"text","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (26, 1, 'cache', 'Cache', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"Catalog Name","id":"catalogname","type":"text","displaytype":"text","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (34, 1, 'DB2', 'DB2', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"Schema Name","id":"catalogname","type":"text","displaytype":"text","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'TC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (36, 1, 'Sybase', 'Sybase', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"Service Name","id":"catalogname","type":"text","displaytype":"text","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":false}]', 'TC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (37, 1, 'JIRA', 'JIRA', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"URL","id":"url","type":"text","displaytype":"text","required":true},{"name":"Project Name","id":"projectname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true}]', 'T', 4, '2020-05-06 12:56:33');
INSERT INTO public.cu_connections VALUES (28, 1, 'Vertica', 'Vertica', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"Host","id":"url","type":"text","displaytype":"text","required":true},{"name":"Port","id":"port","type":"text","displaytype":"number","required":true},{"name":"DB Name","id":"dbname","type":"text","displaytype":"text"},{"name":"Schema Name","id":"catalogname","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password"}]', 'TVC', 1, '2020-02-27 12:56:33');
INSERT INTO public.cu_connections VALUES (38, 1, 'Salesforce', 'Salesforce', '[{"name":"Connection Name","type":"text","id":"connectionname","displaytype":"text","required":true,"pattern":true},{"name":"URL","id":"url","type":"text","displaytype":"text","required":true},{"name":"User Name","id":"dbuser","type":"text","displaytype":"text","required":true},{"name":"Password","id":"dbPassword","type":"text","displaytype":"password","required":true},{"name":"Security Token","id":"securitytoken","type":"text","displaytype":"password","required":true}]', 'T', 4, '2020-04-26 18:44:31.685');
INSERT INTO public.cu_connections VALUES (39, 1, 'Genericjdbc', 'Generic JDBC', '[ {"name":"Connection String","type":"text","id":"connectionstring","displaytype":"text","required":true,"pattern":true}, {"name":"Jar File","type":"browse","allow":".jar","required":true,"id":"jarfile"}, {"name":"Driver Class Name","type":"text","id":"driverclass ","displaytype":"text","required":true,"pattern":true}, {"name":"User Name","type":"text","id":"username ","displaytype":"text","required":true,"pattern":true}, {"name":"Password","type":"text","id":"password ","displaytype":"text","required":true,"pattern":true} ]', 'TC', 1, '2020-04-27 11:45:06.708');
--
-- TOC entry 3709 (class 0 OID 23708)
-- Dependencies: 223
-- Data for Name: cu_dashboard; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3712 (class 0 OID 23721)
-- Dependencies: 226
-- Data for Name: cu_hash_criteria; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3714 (class 0 OID 23730)
-- Dependencies: 228
-- Data for Name: cu_report_visualization; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3715 (class 0 OID 23745)
-- Dependencies: 229
-- Data for Name: cu_report_visualization_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3716 (class 0 OID 23757)
-- Dependencies: 230
-- Data for Name: cu_schedule; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3718 (class 0 OID 23765)
-- Dependencies: 232
-- Data for Name: cu_schedule_log; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3720 (class 0 OID 23773)
-- Dependencies: 234
-- Data for Name: cu_shared_connection_access; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3723 (class 0 OID 23781)
-- Dependencies: 237
-- Data for Name: cu_shared_model; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3724 (class 0 OID 23795)
-- Dependencies: 238
-- Data for Name: cu_shared_model_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3725 (class 0 OID 23802)
-- Dependencies: 239
-- Data for Name: cu_shared_visualization; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3727 (class 0 OID 23817)
-- Dependencies: 241
-- Data for Name: cu_storybook_visualization; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3729 (class 0 OID 23828)
-- Dependencies: 243
-- Data for Name: cu_user_groups; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3732 (class 0 OID 23836)
-- Dependencies: 246
-- Data for Name: data_entity; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3734 (class 0 OID 23848)
-- Dependencies: 248
-- Data for Name: data_hub; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3736 (class 0 OID 23860)
-- Dependencies: 250
-- Data for Name: data_hub_entity; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3737 (class 0 OID 23865)
-- Dependencies: 251
-- Data for Name: data_hub_entity_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3739 (class 0 OID 23873)
-- Dependencies: 253
-- Data for Name: data_model; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3741 (class 0 OID 23885)
-- Dependencies: 255
-- Data for Name: data_model_entity; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3742 (class 0 OID 23895)
-- Dependencies: 256
-- Data for Name: data_model_entity_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3744 (class 0 OID 23904)
-- Dependencies: 258
-- Data for Name: data_model_entity_relation; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3745 (class 0 OID 23911)
-- Dependencies: 259
-- Data for Name: data_model_entity_relation_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3747 (class 0 OID 23919)
-- Dependencies: 261
-- Data for Name: data_model_multifact; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3749 (class 0 OID 23929)
-- Dependencies: 263
-- Data for Name: es_temp; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3751 (class 0 OID 23939)
-- Dependencies: 265
-- Data for Name: export_temp; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3752 (class 0 OID 23958)
-- Dependencies: 266
-- Data for Name: ldap_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3753 (class 0 OID 23973)
-- Dependencies: 267
-- Data for Name: ldap_configurations; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3755 (class 0 OID 23982)
-- Dependencies: 269
-- Data for Name: log_patterns; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.log_patterns VALUES (3, NULL, 'DB LOG', 'dblog', '%{CUSTOMDBDDATE:date} *%{GREEDYDATA:message}', 1, '%{CUSTOMDBDDATE:date} *%{GREEDYDATA:message}', '070917 16:29:12      21 Query       select * from location where id = 1 LIMIT 1', NULL, NULL, NULL, NULL);
INSERT INTO public.log_patterns VALUES (1, NULL, 'APACHE LOG', 'apachelog', '%{IPORHOST:clientip} %{USER:ident} %{USER:auth} \[?%{CUSTOMHTTPDATE:timestamp} %{INT}\]? "(?:%{WORD:verb} %{NOTSPACE:request}(?: HTTP/%{NUMBER:httpversion})?|%{DATA:rawrequest})" %{NUMBER:response} (?:%{NUMBER:bytes}|-)', 1, '%{IPORHOST:clientip} %{USER:ident} %{USER:auth} \[?%{CUSTOMHTTPDATE:timestamp} %{INT:number}\]? "(?:%{WORD:verb} %{NOTSPACE:request}(?: HTTP/%{NUMBER:httpversion})?|%{DATA:rawrequest})" %{NUMBER:response} (?:%{NUMBER:bytes}|-)', '64.242.88.10 - - [07/Mar/2004:16:06:51 -0800] "GET /twiki/bin/rdiff/TWiki/NewUserTemplate?rev1=1.3&rev2=1.2 HTTP/1.1" 200 4523', NULL, NULL, NULL, NULL);
INSERT INTO public.log_patterns VALUES (2, NULL, 'SYS LOG', 'syslog', '%{SYSLOGTIMESTAMP:syslog_timestamp} %{SYSLOGHOST:syslog_hostname} %{DATA:syslog_program}(?:\[%{POSINT:syslog_pid}\])?: %{GREEDYDATA:syslog_message}', 1, '%{SYSLOGTIMESTAMP:syslog_timestamp} %{SYSLOGHOST:syslog_hostname} %{DATA:syslog_program}(?:\[%{POSINT:syslog_pid}\])?: %{GREEDYDATA:syslog_message}', 'Mar 12 12:00:08 server2 rcd[308]: Loaded 12 packages in ''ximian-red-carpet|351'' (0.01878 seconds) ', NULL, NULL, NULL, NULL);


--
-- TOC entry 3757 (class 0 OID 23990)
-- Dependencies: 271
-- Data for Name: login_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3758 (class 0 OID 23993)
-- Dependencies: 272
-- Data for Name: mail_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3760 (class 0 OID 23998)
-- Dependencies: 274
-- Data for Name: ml_models; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.ml_models VALUES (1, 'Clustering', 'clustering', '[{"displayname":"K Value","minval":2,"maxval":20,"fromval":5,"id":"kvalue","displaytype":"number","required":true},{"displayname":"Categorical Encoding","value":[{"value":"AUTO","name":"AUTO"},{"value":"Enum","name":"Enum"},{"value":"OneHotInternal","name":"OneHotInternal"},{"value":"OneHotExplicit","name":"OneHotExplicit"},{"value":"Binary","name":"Binary"},{"value":"Eigen","name":"Eigen"},{"value":"LabelEncoder","name":"LabelEncoder"}],"id":"categorical_encoding","displaytype":"text","required":true},{"displayname":"Ignore Constant Columns","value":[{"value":"1","name":"True"},{"value":"0","name":"False"}],"id":"ignore_const_cols","displaytype":"text","required":true},{"displayname":"Score Each Iteration","value":[{"value":"0","name":"False"},{"value":"1","name":"True"}],"id":"score_each_iteration","displaytype":"text","required":true},{"displayname":"Estimate K","value":[{"value":"1","name":"True"},{"value":"0","name":"False"}],"id":"estimate_k","displaytype":"text","required":true},{"displayname":"Max Iterations","minval":5,"maxval":20,"id":"max_iterations","displaytype":"number","required":true},{"displayname":"Standardize","value":[{"value":"0","name":"False"},{"value":"1","name":"True"}],"id":"standardize","displaytype":"text","required":true},{"displayname":"Max Runtime Seconds","minval":0,"maxval":10,"id":"max_runtime_secs","displaytype":"number","required":true}]', NULL, NULL, 100000);
INSERT INTO public.ml_models VALUES (2, 'Decision Tree', 'decisiontree', '[{"displayname":"Control Parameters","type":"text","name":"controlparameters ","displaytype":"text"},{"displayname":"Method","value":[{"value":"repeatedcv","name":"Repeatedcv"},{"value":"cv","name":"CV"},{"value":"boot","name":"Boot"},{"value":"adaptive_cv","name":"Adaptiv CV"}],"id":"method","displaytype":"text","required":true},{"displayname":"Number","minval":7,"maxval":10,"id":"decisiontreenumber","displaytype":"number","required":true},{"displayname":"Repeats ","minval":2,"maxval":5,"id":"decisiontreerepeats","displaytype":"number","required":true},{"displayname":"Fit parameters","type":"text","name":"fitparameters ","displaytype":"text"},{"displayname":"Split","value":[{"value":"information","name":"Information"}],"id":"split","displaytype":"text","required":true},{"displayname":"Tune Length ","minval":6,"maxval":10,"id":"decisiontreetunelength","displaytype":"number","required":true},{"name":"X Axis","type":"text","id":"decisiontree","displaytype":"string","required":true}]', NULL, NULL, 100000);
INSERT INTO public.ml_models VALUES (5, 'Logistic Regression', 'logisticregression', '[{"name":"X Axis","type":"text","id":"logisticregression","displaytype":"","required":true}]', NULL, NULL, 100000);
INSERT INTO public.ml_models VALUES (6, 'Neural Network', 'neuralnetwork', NULL, NULL, NULL, 100000);
INSERT INTO public.ml_models VALUES (7, 'Text Analysis', 'textanalysis', '[{"displayname":"nstart","minval":5,"maxval":20,"id":"nstart","displaytype":"number","required":true},{"displayname":"burnin","minval":0,"maxval":4000,"id":"burnin","displaytype":"number","required":true},{"displayname":"iter","minval":3,"maxval":20,"id":"iter","displaytype":"number","required":true},{"displayname":"Method","value":[{"value":"Gibbs","name":"Gibbs"},{"value":"VEM","name":"VEM"}],"id":"method","displaytype":"number","required":true},{"displayname":"Best","value":[{"value":"1","name":"True"},{"value":"0","name":"False"}],"id":"best","displaytype":"text","required":true}]', NULL, NULL, 100000);
INSERT INTO public.ml_models VALUES (8, 'Random Forest', 'randomforest', NULL, NULL, NULL, 100000);
INSERT INTO public.ml_models VALUES (4, 'Linear Regression', 'generalizedlinearmodeling', '[{"displayname":"NFolds","minval":5,"maxval":20,"id":"nfolds","displaytype":"number","required":true},{"displayname":"Family","value":[{"value":"gaussian","name":"Gaussian"},{"value":"tweedie","name":"Tweedie"}],"id":"family","displaytype":"number","required":true},{"displayname":"Balance Classes","value":[{"value":"0","name":"False"},{"value":"1","name":"True"}],"id":"balance_classes","displaytype":"text","required":true},{"name":"X Axis","type":"text","id":"generalizedlinearmodeling","displaytype":"number","required":true}]', NULL, NULL, 100000);
INSERT INTO public.ml_models VALUES (3, 'Forecasting', 'forecasting', '[{"displayname":"No Of Periods To Forecast","minval":1,"maxval":50,"id":"periods_forecast","displaytype":"number","required":true},{"displayname":"Model","value":[{"value":"Holtwinters","name":"Holtwinters"},{"value":"ARIMA","name":"ARIMA"}],"id":"forecast_family","displaytype":"text","required":true},{"name":"Y Axis","type":"text","id":"forecasting","displaytype":"number","required":true}]
', NULL, NULL, 100000);


--
-- TOC entry 3762 (class 0 OID 24007)
-- Dependencies: 276
-- Data for Name: ml_temp; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3764 (class 0 OID 24016)
-- Dependencies: 278
-- Data for Name: password_policy_rules; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.password_policy_rules VALUES (1, 1, '[8,16]', 1, 1, 1, 1, 1, 6, 0, 0);


--
-- TOC entry 3765 (class 0 OID 24022)
-- Dependencies: 279
-- Data for Name: securityquestions_user_registration; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.securityquestions_user_registration VALUES (1, 'What Is your favorite book?');
INSERT INTO public.securityquestions_user_registration VALUES (2, 'What is the name of the road you grew up on?');
INSERT INTO public.securityquestions_user_registration VALUES (3, 'What is your mothers maiden name?');
INSERT INTO public.securityquestions_user_registration VALUES (4, 'What was the name of your first/current/favorite pet?');
INSERT INTO public.securityquestions_user_registration VALUES (5, 'What was the first company that you worked for?');
INSERT INTO public.securityquestions_user_registration VALUES (6, 'Where did you meet your spouse?');
INSERT INTO public.securityquestions_user_registration VALUES (7, 'Where did you go to high school/college?');
INSERT INTO public.securityquestions_user_registration VALUES (8, 'What is your favorite food?');
INSERT INTO public.securityquestions_user_registration VALUES (9, 'What city were you born in?');
INSERT INTO public.securityquestions_user_registration VALUES (10, 'Where is your favorite place to vacation?');


--
-- TOC entry 3766 (class 0 OID 24028)
-- Dependencies: 280
-- Data for Name: semantic_names; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3781 (class 0 OID 24587)
-- Dependencies: 295
-- Data for Name: smartinsight_temp; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3768 (class 0 OID 24036)
-- Dependencies: 282
-- Data for Name: storage_views; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3769 (class 0 OID 24042)
-- Dependencies: 283
-- Data for Name: user_account; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.user_account VALUES (2, 1, NULL, 8, NULL, 'Guest', NULL, 'Guest', 1, '5c9c83e88251dc90288910218600b691a446f31e', NULL, NULL, NULL, 'A', NULL, 1, '2018-03-14 14:19:41.003483', NULL, '2015-08-28 15:34:54', NULL, NULL, NULL, NULL, NULL, NULL, '2019-10-11 11:37:25.050332', NULL, NULL, NULL, 0, 0, 0, NULL, 0);
INSERT INTO public.user_account VALUES (1, 1, NULL, 1, NULL, 'Admin', NULL, 'admin', 1, '2bb235d65f3c03dd75a927bc1c988f80c636e0e9', 'admin@admin.com', NULL, NULL, 'A', NULL, 1, '2018-03-14 14:19:41.003483', NULL, '2019-10-11 11:44:43.450395', NULL, NULL, NULL, NULL, NULL, NULL, '2019-10-11 11:37:25.050332', NULL, NULL, NULL, 0, 0, 0, NULL, 0);


--
-- TOC entry 3770 (class 0 OID 24056)
-- Dependencies: 284
-- Data for Name: user_account_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3772 (class 0 OID 24066)
-- Dependencies: 286
-- Data for Name: user_role; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.user_role VALUES (1, 'ROLE_Admin', NULL, 'Admin', NULL, 'A', 1, '2018-03-14 14:18:47.677433', NULL, NULL, NULL);
INSERT INTO public.user_role VALUES (7, 'ROLE_Super_Admin', NULL, 'Super Admin', NULL, 'A', 1, '2018-03-14 14:18:47.677433', NULL, NULL, NULL);
INSERT INTO public.user_role VALUES (8, 'ROLE_Guest', NULL, 'Guest', NULL, 'A', 1, '2018-03-14 14:18:47.677433', NULL, NULL, NULL);
INSERT INTO public.user_role VALUES (9, 'ROLE_Client_Admin', NULL, 'Client Admin', NULL, 'A', 1, '2018-03-14 14:18:47.677433', NULL, NULL, NULL);
INSERT INTO public.user_role VALUES (6, 'ROLE_ReadOnly_User', NULL, 'Explorer', NULL, 'A', 1, '2018-03-14 14:18:47.677433', '2019-05-13 17:26:37.601365', NULL, NULL);
INSERT INTO public.user_role VALUES (4, 'ROLE_Power_User', NULL, 'Creator', NULL, 'A', 1, '2018-03-14 14:18:47.677433', '2019-05-13 17:26:37.601365', NULL, NULL);
INSERT INTO public.user_role VALUES (10, 'ROLE_Viewer', NULL, 'Viewer', NULL, 'A', 1, '2019-05-13 17:26:37.601365', NULL, NULL, NULL);


--
-- TOC entry 3773 (class 0 OID 24073)
-- Dependencies: 287
-- Data for Name: user_role_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3776 (class 0 OID 24085)
-- Dependencies: 290
-- Data for Name: workspace; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3778 (class 0 OID 24097)
-- Dependencies: 292
-- Data for Name: workspace_entity; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3779 (class 0 OID 24108)
-- Dependencies: 293
-- Data for Name: workspace_entity_audit; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3909 (class 0 OID 0)
-- Dependencies: 197
-- Name: bird_reserved_words_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.bird_reserved_words_id', 1, false);


--
-- TOC entry 3910 (class 0 OID 0)
-- Dependencies: 201
-- Name: client_license_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.client_license_id_seq', 1, false);


--
-- TOC entry 3911 (class 0 OID 0)
-- Dependencies: 203
-- Name: client_partner_client_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.client_partner_client_id_seq', 2, false);


--
-- TOC entry 3912 (class 0 OID 0)
-- Dependencies: 205
-- Name: cp_feature_access_cp_feature_access_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cp_feature_access_cp_feature_access_id_seq', 6, true);


--
-- TOC entry 3913 (class 0 OID 0)
-- Dependencies: 207
-- Name: cp_features_feature_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cp_features_feature_id_seq', 122, false);


--
-- TOC entry 3914 (class 0 OID 0)
-- Dependencies: 209
-- Name: cp_groups_group_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cp_groups_group_id_seq', 2, false);


--
-- TOC entry 3915 (class 0 OID 0)
-- Dependencies: 212
-- Name: cu_alert_cu_alert_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_alert_cu_alert_id_seq', 1, false);


--
-- TOC entry 3916 (class 0 OID 0)
-- Dependencies: 214
-- Name: cu_alert_publishinfo_cu_alert_publishinfo_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_alert_publishinfo_cu_alert_publishinfo_id_seq', 1, false);


--
-- TOC entry 3917 (class 0 OID 0)
-- Dependencies: 215
-- Name: cu_alert_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_alert_seq', 1, false);


--
-- TOC entry 3918 (class 0 OID 0)
-- Dependencies: 218
-- Name: cu_connection_access_connection_access_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_connection_access_connection_access_id_seq', 1, false);


--
-- TOC entry 3919 (class 0 OID 0)
-- Dependencies: 220
-- Name: cu_connection_types_type_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_connection_types_type_id_seq', 7, false);


--
-- TOC entry 3920 (class 0 OID 0)
-- Dependencies: 222
-- Name: cu_connections_connections_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_connections_connections_id_seq', 24, true);


--
-- TOC entry 3921 (class 0 OID 0)
-- Dependencies: 224
-- Name: cu_dashboard_cu_dashboard_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_dashboard_cu_dashboard_id_seq', 1, false);


--
-- TOC entry 3922 (class 0 OID 0)
-- Dependencies: 225
-- Name: cu_hash_criteria_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_hash_criteria_id_seq', 26, false);


--
-- TOC entry 3923 (class 0 OID 0)
-- Dependencies: 227
-- Name: cu_report_visualization_cu_report_visualization_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_report_visualization_cu_report_visualization_id_seq', 1, false);


--
-- TOC entry 3924 (class 0 OID 0)
-- Dependencies: 231
-- Name: cu_schedule_cu_schedule_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_schedule_cu_schedule_id_seq', 1, false);


--
-- TOC entry 3925 (class 0 OID 0)
-- Dependencies: 233
-- Name: cu_schedule_log_cu_schedule_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_schedule_log_cu_schedule_log_id_seq', 1, false);


--
-- TOC entry 3926 (class 0 OID 0)
-- Dependencies: 235
-- Name: cu_shared_connection_access_cu_shared_connection_access_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_shared_connection_access_cu_shared_connection_access_id_seq', 1, false);


--
-- TOC entry 3927 (class 0 OID 0)
-- Dependencies: 236
-- Name: cu_shared_model_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_shared_model_id_seq', 1, false);


--
-- TOC entry 3928 (class 0 OID 0)
-- Dependencies: 240
-- Name: cu_shared_visualization_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_shared_visualization_id_seq', 1, false);


--
-- TOC entry 3929 (class 0 OID 0)
-- Dependencies: 242
-- Name: cu_storybook_visualization_storybook_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_storybook_visualization_storybook_id_seq', 1, false);


--
-- TOC entry 3930 (class 0 OID 0)
-- Dependencies: 244
-- Name: cu_user_groups_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cu_user_groups_id_seq', 1, false);


--
-- TOC entry 3931 (class 0 OID 0)
-- Dependencies: 245
-- Name: data_entity_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_entity_id', 1, false);


--
-- TOC entry 3932 (class 0 OID 0)
-- Dependencies: 249
-- Name: data_hub_entity_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_hub_entity_id', 1, false);


--
-- TOC entry 3933 (class 0 OID 0)
-- Dependencies: 247
-- Name: data_hub_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_hub_id', 1, false);


--
-- TOC entry 3934 (class 0 OID 0)
-- Dependencies: 254
-- Name: data_model_entity_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_model_entity_id', 1, false);


--
-- TOC entry 3935 (class 0 OID 0)
-- Dependencies: 257
-- Name: data_model_entity_relation_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_model_entity_relation_id', 1, false);


--
-- TOC entry 3936 (class 0 OID 0)
-- Dependencies: 252
-- Name: data_model_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_model_id', 1, false);


--
-- TOC entry 3937 (class 0 OID 0)
-- Dependencies: 260
-- Name: data_model_mf_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_model_mf_id', 361, false);


--
-- TOC entry 3938 (class 0 OID 0)
-- Dependencies: 262
-- Name: es_temp_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.es_temp_id_seq', 1, false);


--
-- TOC entry 3939 (class 0 OID 0)
-- Dependencies: 264
-- Name: export_temp_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.export_temp_seq', 1, false);


--
-- TOC entry 3940 (class 0 OID 0)
-- Dependencies: 268
-- Name: ldap_configurations_ldap_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ldap_configurations_ldap_id_seq', 1, false);


--
-- TOC entry 3941 (class 0 OID 0)
-- Dependencies: 270
-- Name: log_patterns_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.log_patterns_log_id_seq', 4, false);


--
-- TOC entry 3942 (class 0 OID 0)
-- Dependencies: 273
-- Name: mail_audit_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.mail_audit_id_seq', 1, false);


--
-- TOC entry 3943 (class 0 OID 0)
-- Dependencies: 275
-- Name: ml_temp_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ml_temp_id_seq', 33, false);


--
-- TOC entry 3944 (class 0 OID 0)
-- Dependencies: 277
-- Name: password_policy_rules_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.password_policy_rules_id_seq', 4, false);


--
-- TOC entry 3945 (class 0 OID 0)
-- Dependencies: 281
-- Name: semantic_names_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.semantic_names_id_seq', 1, false);


--
-- TOC entry 3946 (class 0 OID 0)
-- Dependencies: 294
-- Name: smartinsight_temp_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.smartinsight_temp_id_seq', 1, false);


--
-- TOC entry 3947 (class 0 OID 0)
-- Dependencies: 285
-- Name: user_account_user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.user_account_user_id_seq', 3, false);


--
-- TOC entry 3948 (class 0 OID 0)
-- Dependencies: 288
-- Name: user_role_role_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.user_role_role_id_seq', 10, true);


--
-- TOC entry 3949 (class 0 OID 0)
-- Dependencies: 291
-- Name: workspace_entity_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.workspace_entity_id', 1, false);


--
-- TOC entry 3950 (class 0 OID 0)
-- Dependencies: 289
-- Name: workspace_id; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.workspace_id', 1, false);


--
-- TOC entry 3328 (class 2606 OID 24140)
-- Name: client_license client_license_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.client_license
    ADD CONSTRAINT client_license_pkey PRIMARY KEY (id);


--
-- TOC entry 3330 (class 2606 OID 24142)
-- Name: client_partner client_partner_client_admin_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.client_partner
    ADD CONSTRAINT client_partner_client_admin_email_key UNIQUE (client_admin_email);


--
-- TOC entry 3332 (class 2606 OID 24144)
-- Name: client_partner client_partner_client_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.client_partner
    ADD CONSTRAINT client_partner_client_name_key UNIQUE (client_name);


--
-- TOC entry 3334 (class 2606 OID 24146)
-- Name: client_partner client_partner_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.client_partner
    ADD CONSTRAINT client_partner_pkey PRIMARY KEY (client_id);


--
-- TOC entry 3337 (class 2606 OID 24148)
-- Name: cp_feature_access cp_feature_access_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_feature_access
    ADD CONSTRAINT cp_feature_access_pkey PRIMARY KEY (cp_feature_access_id);


--
-- TOC entry 3339 (class 2606 OID 24150)
-- Name: cp_features cp_features_feature_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_features
    ADD CONSTRAINT cp_features_feature_name_key UNIQUE (feature_name);


--
-- TOC entry 3341 (class 2606 OID 24152)
-- Name: cp_features cp_features_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_features
    ADD CONSTRAINT cp_features_pkey PRIMARY KEY (feature_id);


--
-- TOC entry 3343 (class 2606 OID 24154)
-- Name: cp_groups cp_groups_group_name_client_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_groups
    ADD CONSTRAINT cp_groups_group_name_client_id_key UNIQUE (group_name, client_id);


--
-- TOC entry 3345 (class 2606 OID 24156)
-- Name: cp_groups cp_groups_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_groups
    ADD CONSTRAINT cp_groups_pkey PRIMARY KEY (group_id);


--
-- TOC entry 3348 (class 2606 OID 24158)
-- Name: cu_alert cu_alert_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_alert
    ADD CONSTRAINT cu_alert_pkey PRIMARY KEY (cu_alert_id);


--
-- TOC entry 3351 (class 2606 OID 24160)
-- Name: cu_alert_publishinfo cu_alert_publishinfo_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_alert_publishinfo
    ADD CONSTRAINT cu_alert_publishinfo_pkey PRIMARY KEY (cu_alert_publishinfo_id);


--
-- TOC entry 3356 (class 2606 OID 24162)
-- Name: cu_connection_access cu_connection_access_connection_access_name_client_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_access
    ADD CONSTRAINT cu_connection_access_connection_access_name_client_id_key UNIQUE (connection_access_name, client_id);


--
-- TOC entry 3358 (class 2606 OID 24164)
-- Name: cu_connection_access cu_connection_access_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_access
    ADD CONSTRAINT cu_connection_access_pkey PRIMARY KEY (connection_access_id);


--
-- TOC entry 3360 (class 2606 OID 24166)
-- Name: cu_connection_types cu_connection_types_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_types
    ADD CONSTRAINT cu_connection_types_pkey PRIMARY KEY (type_id);


--
-- TOC entry 3364 (class 2606 OID 24168)
-- Name: cu_connections cu_connections_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connections
    ADD CONSTRAINT cu_connections_pkey PRIMARY KEY (connections_id);


--
-- TOC entry 3366 (class 2606 OID 24170)
-- Name: cu_dashboard cu_dashboard_dashboard_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_dashboard
    ADD CONSTRAINT cu_dashboard_dashboard_name_key UNIQUE (dashboard_name);


--
-- TOC entry 3370 (class 2606 OID 24172)
-- Name: cu_dashboard cu_dashboard_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_dashboard
    ADD CONSTRAINT cu_dashboard_pkey PRIMARY KEY (cu_dashboard_id);


--
-- TOC entry 3372 (class 2606 OID 24174)
-- Name: cu_hash_criteria cu_hash_criteria_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_hash_criteria
    ADD CONSTRAINT cu_hash_criteria_pkey PRIMARY KEY (cu_hash_criteria_id);


--
-- TOC entry 3375 (class 2606 OID 24176)
-- Name: cu_report_visualization cu_report_visualization_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_report_visualization
    ADD CONSTRAINT cu_report_visualization_pkey PRIMARY KEY (cu_report_visualization_id);


--
-- TOC entry 3377 (class 2606 OID 24178)
-- Name: cu_report_visualization cu_report_visualization_report_visualization_name_status_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_report_visualization
    ADD CONSTRAINT cu_report_visualization_report_visualization_name_status_key UNIQUE (report_visualization_name, status);


--
-- TOC entry 3383 (class 2606 OID 24180)
-- Name: cu_schedule_log cu_schedule_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_schedule_log
    ADD CONSTRAINT cu_schedule_log_pkey PRIMARY KEY (cu_schedule_log_id);


--
-- TOC entry 3380 (class 2606 OID 24182)
-- Name: cu_schedule cu_schedule_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_schedule
    ADD CONSTRAINT cu_schedule_pkey PRIMARY KEY (cu_schedule_id);


--
-- TOC entry 3386 (class 2606 OID 24184)
-- Name: cu_shared_connection_access cu_shared_connection_access_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_connection_access
    ADD CONSTRAINT cu_shared_connection_access_pkey PRIMARY KEY (cu_shared_connection_access_id);


--
-- TOC entry 3388 (class 2606 OID 24186)
-- Name: cu_shared_model cu_shared_model_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_model
    ADD CONSTRAINT cu_shared_model_pkey PRIMARY KEY (id);


--
-- TOC entry 3390 (class 2606 OID 24188)
-- Name: cu_shared_model cu_shared_model_user_id_client_id_report_visualizat_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_model
    ADD CONSTRAINT cu_shared_model_user_id_client_id_report_visualizat_key UNIQUE (user_id, client_id, model_id, status);


--
-- TOC entry 3395 (class 2606 OID 24190)
-- Name: cu_shared_visualization cu_shared_visualization_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_visualization
    ADD CONSTRAINT cu_shared_visualization_pkey PRIMARY KEY (id);


--
-- TOC entry 3397 (class 2606 OID 24192)
-- Name: cu_shared_visualization cu_shared_visualization_user_id_client_id_report_visualizat_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_visualization
    ADD CONSTRAINT cu_shared_visualization_user_id_client_id_report_visualizat_key UNIQUE (user_id, client_id, report_visualization_id, status);


--
-- TOC entry 3399 (class 2606 OID 24194)
-- Name: cu_storybook_visualization cu_storybook_visualization_id; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_storybook_visualization
    ADD CONSTRAINT cu_storybook_visualization_id PRIMARY KEY (storybook_id);


--
-- TOC entry 3406 (class 2606 OID 24196)
-- Name: cu_user_groups cu_user_groups_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_user_groups
    ADD CONSTRAINT cu_user_groups_id_key UNIQUE (id);


--
-- TOC entry 3408 (class 2606 OID 24198)
-- Name: data_entity data_enity_id_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_entity
    ADD CONSTRAINT data_enity_id_pk PRIMARY KEY (data_entity_id);


--
-- TOC entry 3414 (class 2606 OID 24200)
-- Name: data_hub_entity data_hub_entity_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_hub_entity
    ADD CONSTRAINT data_hub_entity_pk PRIMARY KEY (data_hub_entity_id);


--
-- TOC entry 3410 (class 2606 OID 24202)
-- Name: data_hub data_hub_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_hub
    ADD CONSTRAINT data_hub_pkey PRIMARY KEY (data_hub_id);


--
-- TOC entry 3423 (class 2606 OID 24204)
-- Name: data_model_entity data_model_entity_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity
    ADD CONSTRAINT data_model_entity_pkey PRIMARY KEY (data_model_entity_id);


--
-- TOC entry 3425 (class 2606 OID 24206)
-- Name: data_model_entity_relation data_model_entity_relation_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity_relation
    ADD CONSTRAINT data_model_entity_relation_pkey PRIMARY KEY (data_model_entity_relation_id);


--
-- TOC entry 3427 (class 2606 OID 24208)
-- Name: data_model_multifact data_model_multifact_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_multifact
    ADD CONSTRAINT data_model_multifact_pkey PRIMARY KEY (data_model_mf_id);


--
-- TOC entry 3419 (class 2606 OID 24210)
-- Name: data_model data_model_name; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model
    ADD CONSTRAINT data_model_name UNIQUE (name);


--
-- TOC entry 3421 (class 2606 OID 24212)
-- Name: data_model data_model_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model
    ADD CONSTRAINT data_model_pkey PRIMARY KEY (data_model_id);


--
-- TOC entry 3429 (class 2606 OID 24214)
-- Name: es_temp es_temp_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.es_temp
    ADD CONSTRAINT es_temp_id_key UNIQUE (id);


--
-- TOC entry 3431 (class 2606 OID 24216)
-- Name: export_temp export_temp_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.export_temp
    ADD CONSTRAINT export_temp_pkey PRIMARY KEY (export_id);


--
-- TOC entry 3434 (class 2606 OID 24218)
-- Name: ldap_configurations ldap_configurations_ldap_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ldap_configurations
    ADD CONSTRAINT ldap_configurations_ldap_id_key UNIQUE (ldap_id);


--
-- TOC entry 3436 (class 2606 OID 24220)
-- Name: ldap_configurations ldap_configurations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ldap_configurations
    ADD CONSTRAINT ldap_configurations_pkey PRIMARY KEY (ldap_name, client_id);


--
-- TOC entry 3439 (class 2606 OID 24222)
-- Name: log_patterns log_patterns_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.log_patterns
    ADD CONSTRAINT log_patterns_pkey PRIMARY KEY (log_id);


--
-- TOC entry 3441 (class 2606 OID 24224)
-- Name: mail_audit mail_audit_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mail_audit
    ADD CONSTRAINT mail_audit_pkey PRIMARY KEY (id);


--
-- TOC entry 3443 (class 2606 OID 24226)
-- Name: ml_temp ml_temp_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ml_temp
    ADD CONSTRAINT ml_temp_id_key UNIQUE (id);


--
-- TOC entry 3445 (class 2606 OID 24228)
-- Name: password_policy_rules password_policy_rules_client_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.password_policy_rules
    ADD CONSTRAINT password_policy_rules_client_id_key UNIQUE (client_id);


--
-- TOC entry 3447 (class 2606 OID 24230)
-- Name: password_policy_rules password_policy_rules_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.password_policy_rules
    ADD CONSTRAINT password_policy_rules_pkey PRIMARY KEY (id);


--
-- TOC entry 3449 (class 2606 OID 24232)
-- Name: securityquestions_user_registration securityquestions_user_registration_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.securityquestions_user_registration
    ADD CONSTRAINT securityquestions_user_registration_pkey PRIMARY KEY (question_id);


--
-- TOC entry 3451 (class 2606 OID 24234)
-- Name: semantic_names semantic_names_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.semantic_names
    ADD CONSTRAINT semantic_names_pkey PRIMARY KEY (id);


--
-- TOC entry 3471 (class 2606 OID 24595)
-- Name: smartinsight_temp smartinsight_temp_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.smartinsight_temp
    ADD CONSTRAINT smartinsight_temp_id_key UNIQUE (id);


--
-- TOC entry 3401 (class 2606 OID 24236)
-- Name: cu_storybook_visualization storybook_name_unq; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_storybook_visualization
    ADD CONSTRAINT storybook_name_unq UNIQUE (storybook_name);


--
-- TOC entry 3456 (class 2606 OID 24238)
-- Name: user_account user_account_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_account
    ADD CONSTRAINT user_account_pkey PRIMARY KEY (user_id);


--
-- TOC entry 3459 (class 2606 OID 24240)
-- Name: user_account user_account_user_login_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_account
    ADD CONSTRAINT user_account_user_login_name_key UNIQUE (user_login_name);


--
-- TOC entry 3461 (class 2606 OID 24242)
-- Name: user_role user_role_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_role
    ADD CONSTRAINT user_role_pkey PRIMARY KEY (role_id);


--
-- TOC entry 3463 (class 2606 OID 24244)
-- Name: user_role user_role_role_name_client_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_role
    ADD CONSTRAINT user_role_role_name_client_id_key UNIQUE (role_name, client_id);


--
-- TOC entry 3469 (class 2606 OID 24246)
-- Name: workspace_entity workspace_entity_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace_entity
    ADD CONSTRAINT workspace_entity_pkey PRIMARY KEY (workspace_entity_id);


--
-- TOC entry 3465 (class 2606 OID 24248)
-- Name: workspace workspace_id_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace
    ADD CONSTRAINT workspace_id_pk PRIMARY KEY (workspace_id);


--
-- TOC entry 3467 (class 2606 OID 24250)
-- Name: workspace workspace_name; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace
    ADD CONSTRAINT workspace_name UNIQUE (workspace_name);


--
-- TOC entry 3335 (class 1259 OID 24251)
-- Name: cp_feature_access_fk_cpf_ur; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cp_feature_access_fk_cpf_ur ON public.cp_feature_access USING btree (role_id);


--
-- TOC entry 3346 (class 1259 OID 24252)
-- Name: cu_alert_FK_reportId; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_alert_FK_reportId" ON public.cu_alert USING btree (report_visualization_id);


--
-- TOC entry 3349 (class 1259 OID 24253)
-- Name: cu_alert_publishinfo_FK_alertId; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_alert_publishinfo_FK_alertId" ON public.cu_alert_publishinfo USING btree (alert_id);


--
-- TOC entry 3352 (class 1259 OID 24254)
-- Name: cu_connection_access_FK_cu_connection_access_client_partner; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_connection_access_FK_cu_connection_access_client_partner" ON public.cu_connection_access USING btree (client_id);


--
-- TOC entry 3353 (class 1259 OID 24255)
-- Name: cu_connection_access_FK_cu_connection_access_cu_connections; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_connection_access_FK_cu_connection_access_cu_connections" ON public.cu_connection_access USING btree (connection_id);


--
-- TOC entry 3354 (class 1259 OID 24256)
-- Name: cu_connection_access_FK_cu_connection_access_user_account; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_connection_access_FK_cu_connection_access_user_account" ON public.cu_connection_access USING btree (user_id);


--
-- TOC entry 3361 (class 1259 OID 24257)
-- Name: cu_connections_FK_cu_connections_client_partner; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_connections_FK_cu_connections_client_partner" ON public.cu_connections USING btree (client_id);


--
-- TOC entry 3362 (class 1259 OID 24258)
-- Name: cu_connections_FK_cu_connections_cu_connection_types; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_connections_FK_cu_connections_cu_connection_types" ON public.cu_connections USING btree (connection_type_id);


--
-- TOC entry 3367 (class 1259 OID 24259)
-- Name: cu_dashboard_fk_report_viz_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_dashboard_fk_report_viz_id ON public.cu_dashboard USING btree (cu_report_visualization_id);


--
-- TOC entry 3368 (class 1259 OID 24260)
-- Name: cu_dashboard_fk_ua_cud_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_dashboard_fk_ua_cud_idx ON public.cu_dashboard USING btree (user_id);


--
-- TOC entry 3373 (class 1259 OID 24261)
-- Name: cu_report_visualization_fk_ua_curv; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_report_visualization_fk_ua_curv ON public.cu_report_visualization USING btree (user_id);


--
-- TOC entry 3378 (class 1259 OID 24262)
-- Name: cu_schedule_FK_cu_schedule_cu_report_visualization; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_schedule_FK_cu_schedule_cu_report_visualization" ON public.cu_schedule USING btree (cu_report_visualization_id);


--
-- TOC entry 3381 (class 1259 OID 24263)
-- Name: cu_schedule_log_FK_cu_schedule_log_cu_schedule; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "cu_schedule_log_FK_cu_schedule_log_cu_schedule" ON public.cu_schedule_log USING btree (cu_schedule_id);


--
-- TOC entry 3384 (class 1259 OID 24264)
-- Name: cu_shared_connection_access_fk_cus_cusa_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_shared_connection_access_fk_cus_cusa_idx ON public.cu_shared_connection_access USING btree (cu_connection_access_id);


--
-- TOC entry 3391 (class 1259 OID 24265)
-- Name: cu_shared_visualization_fk_client_partner; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_shared_visualization_fk_client_partner ON public.cu_shared_visualization USING btree (client_id);


--
-- TOC entry 3392 (class 1259 OID 24266)
-- Name: cu_shared_visualization_fk_report_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_shared_visualization_fk_report_id ON public.cu_shared_visualization USING btree (report_visualization_id);


--
-- TOC entry 3393 (class 1259 OID 24267)
-- Name: cu_shared_visualization_fk_user_account; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_shared_visualization_fk_user_account ON public.cu_shared_visualization USING btree (user_id);


--
-- TOC entry 3402 (class 1259 OID 24268)
-- Name: cu_user_groups_cu_user_groups_client_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_user_groups_cu_user_groups_client_id ON public.cu_user_groups USING btree (client_id);


--
-- TOC entry 3403 (class 1259 OID 24269)
-- Name: cu_user_groups_cu_user_groups_group_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_user_groups_cu_user_groups_group_id ON public.cu_user_groups USING btree (group_id);


--
-- TOC entry 3404 (class 1259 OID 24270)
-- Name: cu_user_groups_cu_user_groups_pk; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX cu_user_groups_cu_user_groups_pk ON public.cu_user_groups USING btree (user_id, group_id, client_id, status);


--
-- TOC entry 3411 (class 1259 OID 24271)
-- Name: fki_client_id_fk; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX fki_client_id_fk ON public.data_hub USING btree (client_id);


--
-- TOC entry 3415 (class 1259 OID 24272)
-- Name: fki_data_entity_fk; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX fki_data_entity_fk ON public.data_hub_entity USING btree (data_hub_entity_id);


--
-- TOC entry 3416 (class 1259 OID 24273)
-- Name: fki_data_entity_id_fk; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX fki_data_entity_id_fk ON public.data_hub_entity USING btree (entity_id);


--
-- TOC entry 3417 (class 1259 OID 24274)
-- Name: fki_data_hub_id_fk; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX fki_data_hub_id_fk ON public.data_hub_entity USING btree (data_hub_id);


--
-- TOC entry 3412 (class 1259 OID 24275)
-- Name: fki_user_id_fk; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX fki_user_id_fk ON public.data_hub USING btree (user_id);


--
-- TOC entry 3432 (class 1259 OID 24276)
-- Name: ldap_configurations_FK__client_partner; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "ldap_configurations_FK__client_partner" ON public.ldap_configurations USING btree (client_id);


--
-- TOC entry 3437 (class 1259 OID 24277)
-- Name: log_patterns_fk_log_userid; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX log_patterns_fk_log_userid ON public.log_patterns USING btree (created_by);


--
-- TOC entry 3452 (class 1259 OID 24278)
-- Name: semantic_names_semantic_schema_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX semantic_names_semantic_schema_id ON public.semantic_names USING btree (cu_schema_id);


--
-- TOC entry 3453 (class 1259 OID 24279)
-- Name: user_account_FK_user_account_ldap_configurations; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX "user_account_FK_user_account_ldap_configurations" ON public.user_account USING btree (ldap_id);


--
-- TOC entry 3454 (class 1259 OID 24280)
-- Name: user_account_fk_ua_cp_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX user_account_fk_ua_cp_idx ON public.user_account USING btree (client_id);


--
-- TOC entry 3457 (class 1259 OID 24281)
-- Name: user_account_role_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX user_account_role_id ON public.user_account USING btree (role_id);


--
-- TOC entry 3525 (class 2620 OID 24282)
-- Name: client_partner client_partner_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER client_partner_before_insert BEFORE INSERT ON public.client_partner FOR EACH ROW EXECUTE PROCEDURE public.client_partner_before_insert();


--
-- TOC entry 3526 (class 2620 OID 24283)
-- Name: client_partner client_partner_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER client_partner_before_update BEFORE UPDATE ON public.client_partner FOR EACH ROW EXECUTE PROCEDURE public.client_partner_before_update();


--
-- TOC entry 3527 (class 2620 OID 24284)
-- Name: cp_feature_access cp_feature_access_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cp_feature_access_before_insert BEFORE INSERT ON public.cp_feature_access FOR EACH ROW EXECUTE PROCEDURE public.cp_feature_access_before_insert();


--
-- TOC entry 3528 (class 2620 OID 24285)
-- Name: cp_feature_access cp_feature_access_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cp_feature_access_before_update BEFORE UPDATE ON public.cp_feature_access FOR EACH ROW EXECUTE PROCEDURE public.cp_feature_access_before_update();


--
-- TOC entry 3529 (class 2620 OID 24286)
-- Name: cp_groups cp_group_after_update_for_updating_user_roleid; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cp_group_after_update_for_updating_user_roleid AFTER UPDATE ON public.cp_groups FOR EACH ROW EXECUTE PROCEDURE public.cp_group_after_update_for_updating_user_roleid();


--
-- TOC entry 3530 (class 2620 OID 24287)
-- Name: cp_groups cp_groups_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cp_groups_before_insert BEFORE INSERT ON public.cp_groups FOR EACH ROW EXECUTE PROCEDURE public.cp_groups_before_insert();


--
-- TOC entry 3531 (class 2620 OID 24288)
-- Name: cp_groups cp_groups_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cp_groups_before_update BEFORE UPDATE ON public.cp_groups FOR EACH ROW EXECUTE PROCEDURE public.cp_groups_before_update();


--
-- TOC entry 3532 (class 2620 OID 24289)
-- Name: cu_connection_access cu_connection_access_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_connection_access_before_update BEFORE UPDATE ON public.cu_connection_access FOR EACH ROW EXECUTE PROCEDURE public.cu_connection_access_before_update();


--
-- TOC entry 3533 (class 2620 OID 24290)
-- Name: cu_dashboard cu_dashboard_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_dashboard_before_insert BEFORE INSERT ON public.cu_dashboard FOR EACH ROW EXECUTE PROCEDURE public.cu_dashboard_before_insert();


--
-- TOC entry 3534 (class 2620 OID 24291)
-- Name: cu_dashboard cu_dashboard_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_dashboard_before_update AFTER UPDATE ON public.cu_dashboard FOR EACH ROW EXECUTE PROCEDURE public.cu_dashboard_before_update();


--
-- TOC entry 3535 (class 2620 OID 24292)
-- Name: cu_report_visualization cu_report_visualization_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_report_visualization_before_insert BEFORE INSERT ON public.cu_report_visualization FOR EACH ROW EXECUTE PROCEDURE public.cu_report_visualization_before_insert();


--
-- TOC entry 3536 (class 2620 OID 24293)
-- Name: cu_report_visualization cu_report_visualization_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_report_visualization_before_update BEFORE UPDATE ON public.cu_report_visualization FOR EACH ROW EXECUTE PROCEDURE public.cu_report_visualization_before_update();


--
-- TOC entry 3537 (class 2620 OID 24294)
-- Name: cu_schedule cu_schedule_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_schedule_before_insert BEFORE INSERT ON public.cu_schedule FOR EACH ROW EXECUTE PROCEDURE public.cu_schedule_before_insert();


--
-- TOC entry 3538 (class 2620 OID 24295)
-- Name: cu_schedule cu_schedule_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_schedule_before_update BEFORE UPDATE ON public.cu_schedule FOR EACH ROW EXECUTE PROCEDURE public.cu_schedule_before_update();


--
-- TOC entry 3539 (class 2620 OID 24296)
-- Name: cu_schedule_log cu_schedule_log_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_schedule_log_before_insert BEFORE INSERT ON public.cu_schedule_log FOR EACH ROW EXECUTE PROCEDURE public.cu_schedule_log_before_insert();


--
-- TOC entry 3540 (class 2620 OID 24297)
-- Name: cu_schedule_log cu_schedule_log_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_schedule_log_before_update BEFORE UPDATE ON public.cu_schedule_log FOR EACH ROW EXECUTE PROCEDURE public.cu_schedule_log_before_update();


--
-- TOC entry 3541 (class 2620 OID 24298)
-- Name: cu_shared_connection_access cu_shared_connection_access_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_shared_connection_access_before_insert BEFORE INSERT ON public.cu_shared_connection_access FOR EACH ROW EXECUTE PROCEDURE public.cu_shared_connection_access_before_insert();


--
-- TOC entry 3542 (class 2620 OID 24299)
-- Name: cu_shared_model cu_shared_model_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_shared_model_before_update BEFORE UPDATE ON public.cu_shared_model FOR EACH ROW EXECUTE PROCEDURE public.cu_shared_visualization_before_update();


--
-- TOC entry 3543 (class 2620 OID 24300)
-- Name: cu_shared_visualization cu_shared_visualization_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_shared_visualization_before_insert BEFORE INSERT ON public.cu_shared_visualization FOR EACH ROW EXECUTE PROCEDURE public.cu_shared_visualization_before_insert();


--
-- TOC entry 3544 (class 2620 OID 24301)
-- Name: cu_shared_visualization cu_shared_visualization_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_shared_visualization_before_update BEFORE UPDATE ON public.cu_shared_visualization FOR EACH ROW EXECUTE PROCEDURE public.cu_shared_visualization_before_update();


--
-- TOC entry 3545 (class 2620 OID 24302)
-- Name: cu_user_groups cu_user_groups_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_user_groups_before_insert BEFORE INSERT ON public.cu_user_groups FOR EACH ROW EXECUTE PROCEDURE public.cu_user_groups_before_insert();


--
-- TOC entry 3546 (class 2620 OID 24303)
-- Name: cu_user_groups cu_user_groups_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER cu_user_groups_before_update BEFORE UPDATE ON public.cu_user_groups FOR EACH ROW EXECUTE PROCEDURE public.cu_user_groups_before_update();


--
-- TOC entry 3547 (class 2620 OID 24304)
-- Name: data_entity data_entity_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER data_entity_before_update BEFORE UPDATE ON public.data_entity FOR EACH ROW EXECUTE PROCEDURE public.data_entity_before_update();


--
-- TOC entry 3548 (class 2620 OID 24305)
-- Name: data_hub data_hub_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER data_hub_before_update BEFORE UPDATE ON public.data_hub FOR EACH ROW EXECUTE PROCEDURE public.data_hub_before_update();


--
-- TOC entry 3549 (class 2620 OID 24306)
-- Name: data_model data_model_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER data_model_before_update BEFORE UPDATE ON public.data_model FOR EACH ROW EXECUTE PROCEDURE public.data_model_before_update();


--
-- TOC entry 3550 (class 2620 OID 24307)
-- Name: data_model_entity data_model_entity_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER data_model_entity_before_update BEFORE UPDATE ON public.data_model_entity FOR EACH ROW EXECUTE PROCEDURE public.data_model_entity_before_update();


--
-- TOC entry 3551 (class 2620 OID 24308)
-- Name: es_temp es_temp_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER es_temp_before_insert BEFORE INSERT ON public.es_temp FOR EACH ROW EXECUTE PROCEDURE public.es_temp_before_insert();


--
-- TOC entry 3552 (class 2620 OID 24309)
-- Name: es_temp es_temp_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER es_temp_before_update BEFORE UPDATE ON public.es_temp FOR EACH ROW EXECUTE PROCEDURE public.es_temp_before_update();


--
-- TOC entry 3553 (class 2620 OID 24310)
-- Name: ldap_configurations ldap_configurations_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ldap_configurations_before_insert BEFORE UPDATE ON public.ldap_configurations FOR EACH ROW EXECUTE PROCEDURE public.ldap_configurations_before_insert();


--
-- TOC entry 3554 (class 2620 OID 24311)
-- Name: ldap_configurations ldap_configurations_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER ldap_configurations_before_update BEFORE UPDATE ON public.ldap_configurations FOR EACH ROW EXECUTE PROCEDURE public.ldap_configurations_before_update();


--
-- TOC entry 3555 (class 2620 OID 24312)
-- Name: user_account user_account_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER user_account_before_insert BEFORE INSERT ON public.user_account FOR EACH ROW EXECUTE PROCEDURE public.user_account_before_insert();


--
-- TOC entry 3556 (class 2620 OID 24313)
-- Name: user_account user_account_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER user_account_before_update BEFORE UPDATE ON public.user_account FOR EACH ROW EXECUTE PROCEDURE public.user_account_before_update();


--
-- TOC entry 3557 (class 2620 OID 24314)
-- Name: user_role user_role_before_insert; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER user_role_before_insert BEFORE INSERT ON public.user_role FOR EACH ROW EXECUTE PROCEDURE public.user_role_before_insert();


--
-- TOC entry 3558 (class 2620 OID 24315)
-- Name: user_role user_role_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER user_role_before_update BEFORE UPDATE ON public.user_role FOR EACH ROW EXECUTE PROCEDURE public.user_role_before_update();


--
-- TOC entry 3559 (class 2620 OID 24316)
-- Name: workspace workspace_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER workspace_before_update BEFORE UPDATE ON public.workspace FOR EACH ROW EXECUTE PROCEDURE public.workspace_before_update();


--
-- TOC entry 3560 (class 2620 OID 24317)
-- Name: workspace_entity workspace_entity_before_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER workspace_entity_before_update BEFORE UPDATE ON public.workspace_entity FOR EACH ROW EXECUTE PROCEDURE public.workspace_entity_before_update();


--
-- TOC entry 3512 (class 2606 OID 24318)
-- Name: ldap_configurations FK__client_partner; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ldap_configurations
    ADD CONSTRAINT "FK__client_partner" FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3473 (class 2606 OID 24323)
-- Name: cu_alert_publishinfo FK_alertId; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_alert_publishinfo
    ADD CONSTRAINT "FK_alertId" FOREIGN KEY (alert_id) REFERENCES public.cu_alert(cu_alert_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3474 (class 2606 OID 24328)
-- Name: cu_connection_access FK_cu_connection_access_client_partner; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_access
    ADD CONSTRAINT "FK_cu_connection_access_client_partner" FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3475 (class 2606 OID 24333)
-- Name: cu_connection_access FK_cu_connection_access_cu_connections; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_access
    ADD CONSTRAINT "FK_cu_connection_access_cu_connections" FOREIGN KEY (connection_id) REFERENCES public.cu_connections(connections_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3476 (class 2606 OID 24338)
-- Name: cu_connection_access FK_cu_connection_access_user_account; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connection_access
    ADD CONSTRAINT "FK_cu_connection_access_user_account" FOREIGN KEY (user_id) REFERENCES public.user_account(user_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3477 (class 2606 OID 24343)
-- Name: cu_connections FK_cu_connections_client_partner; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connections
    ADD CONSTRAINT "FK_cu_connections_client_partner" FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3478 (class 2606 OID 24348)
-- Name: cu_connections FK_cu_connections_cu_connection_types; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_connections
    ADD CONSTRAINT "FK_cu_connections_cu_connection_types" FOREIGN KEY (connection_type_id) REFERENCES public.cu_connection_types(type_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3482 (class 2606 OID 24353)
-- Name: cu_schedule_log FK_cu_schedule_log_cu_schedule; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_schedule_log
    ADD CONSTRAINT "FK_cu_schedule_log_cu_schedule" FOREIGN KEY (cu_schedule_id) REFERENCES public.cu_schedule(cu_schedule_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3515 (class 2606 OID 24358)
-- Name: user_account FK_user_account_ldap_configurations; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_account
    ADD CONSTRAINT "FK_user_account_ldap_configurations" FOREIGN KEY (ldap_id) REFERENCES public.ldap_configurations(ldap_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3494 (class 2606 OID 24363)
-- Name: data_hub client_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_hub
    ADD CONSTRAINT client_id_fk FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3491 (class 2606 OID 24368)
-- Name: data_entity client_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_entity
    ADD CONSTRAINT client_id_fk FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3518 (class 2606 OID 24373)
-- Name: workspace client_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace
    ADD CONSTRAINT client_id_fk FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3521 (class 2606 OID 24378)
-- Name: workspace_entity client_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace_entity
    ADD CONSTRAINT client_id_fk FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3498 (class 2606 OID 24383)
-- Name: data_model client_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model
    ADD CONSTRAINT client_id_fk FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3501 (class 2606 OID 24388)
-- Name: data_model_entity client_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity
    ADD CONSTRAINT client_id_fk FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3509 (class 2606 OID 24393)
-- Name: data_model_multifact client_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_multifact
    ADD CONSTRAINT client_id_fk FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3492 (class 2606 OID 24398)
-- Name: data_entity connection_access_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_entity
    ADD CONSTRAINT connection_access_id_fk FOREIGN KEY (connection_access_id) REFERENCES public.cu_connection_access(connection_access_id);


--
-- TOC entry 3522 (class 2606 OID 24403)
-- Name: workspace_entity connection_access_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace_entity
    ADD CONSTRAINT connection_access_id_fk FOREIGN KEY (connection_access_id) REFERENCES public.cu_connection_access(connection_access_id);


--
-- TOC entry 3502 (class 2606 OID 24408)
-- Name: data_model_entity connection_access_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity
    ADD CONSTRAINT connection_access_id_fk FOREIGN KEY (connection_access_id) REFERENCES public.cu_connection_access(connection_access_id);


--
-- TOC entry 3488 (class 2606 OID 24413)
-- Name: cu_user_groups cu_user_groups_client_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_user_groups
    ADD CONSTRAINT cu_user_groups_client_id FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3489 (class 2606 OID 24418)
-- Name: cu_user_groups cu_user_groups_group_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_user_groups
    ADD CONSTRAINT cu_user_groups_group_id FOREIGN KEY (group_id) REFERENCES public.cp_groups(group_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3490 (class 2606 OID 24423)
-- Name: cu_user_groups cu_user_groups_user_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_user_groups
    ADD CONSTRAINT cu_user_groups_user_id FOREIGN KEY (user_id) REFERENCES public.user_account(user_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3496 (class 2606 OID 24428)
-- Name: data_hub_entity data_entity_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_hub_entity
    ADD CONSTRAINT data_entity_fk FOREIGN KEY (entity_id) REFERENCES public.data_entity(data_entity_id);


--
-- TOC entry 3497 (class 2606 OID 24433)
-- Name: data_hub_entity data_hub_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_hub_entity
    ADD CONSTRAINT data_hub_id_fk FOREIGN KEY (data_hub_id) REFERENCES public.data_hub(data_hub_id);


--
-- TOC entry 3506 (class 2606 OID 24438)
-- Name: data_model_entity_relation data_model_entity_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity_relation
    ADD CONSTRAINT data_model_entity_id_fk FOREIGN KEY (data_model_entity_id) REFERENCES public.data_model_entity(data_model_entity_id);


--
-- TOC entry 3503 (class 2606 OID 24443)
-- Name: data_model_entity data_model_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity
    ADD CONSTRAINT data_model_id_fk FOREIGN KEY (data_model_id) REFERENCES public.data_model(data_model_id);


--
-- TOC entry 3507 (class 2606 OID 24448)
-- Name: data_model_entity_relation data_model_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity_relation
    ADD CONSTRAINT data_model_id_fk FOREIGN KEY (data_model_id) REFERENCES public.data_model(data_model_id);


--
-- TOC entry 3511 (class 2606 OID 24453)
-- Name: es_temp data_model_id_tempFK; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.es_temp
    ADD CONSTRAINT "data_model_id_tempFK" FOREIGN KEY (data_model_id) REFERENCES public.data_model(data_model_id);


--
-- TOC entry 3486 (class 2606 OID 24458)
-- Name: cu_shared_visualization fk_client_partner; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_visualization
    ADD CONSTRAINT fk_client_partner FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3484 (class 2606 OID 24463)
-- Name: cu_shared_model fk_client_partner; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_model
    ADD CONSTRAINT fk_client_partner FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3472 (class 2606 OID 24468)
-- Name: cp_feature_access fk_cpf_ur; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cp_feature_access
    ADD CONSTRAINT fk_cpf_ur FOREIGN KEY (role_id) REFERENCES public.user_role(role_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3483 (class 2606 OID 24473)
-- Name: cu_shared_connection_access fk_cua_cusc; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_connection_access
    ADD CONSTRAINT fk_cua_cusc FOREIGN KEY (cu_connection_access_id) REFERENCES public.cu_connection_access(connection_access_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3513 (class 2606 OID 24478)
-- Name: log_patterns fk_log_userid; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.log_patterns
    ADD CONSTRAINT fk_log_userid FOREIGN KEY (created_by) REFERENCES public.user_account(user_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3516 (class 2606 OID 24483)
-- Name: user_account fk_ua_cp; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_account
    ADD CONSTRAINT fk_ua_cp FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3514 (class 2606 OID 24488)
-- Name: password_policy_rules fk_ua_cp; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.password_policy_rules
    ADD CONSTRAINT fk_ua_cp FOREIGN KEY (client_id) REFERENCES public.client_partner(client_id);


--
-- TOC entry 3479 (class 2606 OID 24493)
-- Name: cu_dashboard fk_ua_cud; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_dashboard
    ADD CONSTRAINT fk_ua_cud FOREIGN KEY (user_id) REFERENCES public.user_account(user_id);


--
-- TOC entry 3480 (class 2606 OID 24498)
-- Name: cu_report_visualization fk_ua_curv; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_report_visualization
    ADD CONSTRAINT fk_ua_curv FOREIGN KEY (user_id) REFERENCES public.user_account(user_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3517 (class 2606 OID 24503)
-- Name: user_account fk_ua_ur; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_account
    ADD CONSTRAINT fk_ua_ur FOREIGN KEY (role_id) REFERENCES public.user_role(role_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3487 (class 2606 OID 24508)
-- Name: cu_shared_visualization fk_user_account; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_visualization
    ADD CONSTRAINT fk_user_account FOREIGN KEY (user_id) REFERENCES public.user_account(user_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3485 (class 2606 OID 24513)
-- Name: cu_shared_model fk_user_account; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_shared_model
    ADD CONSTRAINT fk_user_account FOREIGN KEY (user_id) REFERENCES public.user_account(user_id) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- TOC entry 3481 (class 2606 OID 24518)
-- Name: cu_report_visualization fk_viz_datamodel; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cu_report_visualization
    ADD CONSTRAINT fk_viz_datamodel FOREIGN KEY (data_model_id) REFERENCES public.data_model(data_model_id);


--
-- TOC entry 3519 (class 2606 OID 24523)
-- Name: workspace hub_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace
    ADD CONSTRAINT hub_id_fk FOREIGN KEY (hub_id) REFERENCES public.data_hub(data_hub_id);


--
-- TOC entry 3508 (class 2606 OID 24528)
-- Name: data_model_entity_relation parent_entity_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity_relation
    ADD CONSTRAINT parent_entity_id_fk FOREIGN KEY (parent_entity_id) REFERENCES public.data_model_entity(data_model_entity_id);


--
-- TOC entry 3495 (class 2606 OID 24533)
-- Name: data_hub user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_hub
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.user_account(user_id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3493 (class 2606 OID 24538)
-- Name: data_entity user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_entity
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.user_account(user_id);


--
-- TOC entry 3520 (class 2606 OID 24543)
-- Name: workspace user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.user_account(user_id);


--
-- TOC entry 3523 (class 2606 OID 24548)
-- Name: workspace_entity user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace_entity
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.user_account(user_id);


--
-- TOC entry 3499 (class 2606 OID 24553)
-- Name: data_model user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.user_account(user_id);


--
-- TOC entry 3504 (class 2606 OID 24558)
-- Name: data_model_entity user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.user_account(user_id);


--
-- TOC entry 3510 (class 2606 OID 24563)
-- Name: data_model_multifact user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_multifact
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.user_account(user_id);


--
-- TOC entry 3505 (class 2606 OID 24568)
-- Name: data_model_entity workspace_entity_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model_entity
    ADD CONSTRAINT workspace_entity_id_fk FOREIGN KEY (workspace_entity_id) REFERENCES public.workspace_entity(workspace_entity_id);


--
-- TOC entry 3524 (class 2606 OID 24573)
-- Name: workspace_entity workspace_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workspace_entity
    ADD CONSTRAINT workspace_id_fk FOREIGN KEY (workspace_id) REFERENCES public.workspace(workspace_id);


--
-- TOC entry 3500 (class 2606 OID 24578)
-- Name: data_model workspace_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_model
    ADD CONSTRAINT workspace_id_fk FOREIGN KEY (workspace_id) REFERENCES public.workspace(workspace_id);


--
-- TOC entry 3787 (class 0 OID 0)
-- Dependencies: 196
-- Name: TABLE api_table; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.api_table TO PUBLIC;


--
-- TOC entry 3788 (class 0 OID 0)
-- Dependencies: 199
-- Name: TABLE client_license; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.client_license TO PUBLIC;


--
-- TOC entry 3789 (class 0 OID 0)
-- Dependencies: 200
-- Name: TABLE client_license_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.client_license_audit TO PUBLIC;


--
-- TOC entry 3791 (class 0 OID 0)
-- Dependencies: 201
-- Name: SEQUENCE client_license_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.client_license_id_seq TO PUBLIC;


--
-- TOC entry 3792 (class 0 OID 0)
-- Dependencies: 202
-- Name: TABLE client_partner; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.client_partner TO PUBLIC;


--
-- TOC entry 3794 (class 0 OID 0)
-- Dependencies: 203
-- Name: SEQUENCE client_partner_client_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.client_partner_client_id_seq TO PUBLIC;


--
-- TOC entry 3795 (class 0 OID 0)
-- Dependencies: 204
-- Name: TABLE cp_feature_access; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cp_feature_access TO PUBLIC;


--
-- TOC entry 3797 (class 0 OID 0)
-- Dependencies: 205
-- Name: SEQUENCE cp_feature_access_cp_feature_access_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cp_feature_access_cp_feature_access_id_seq TO PUBLIC;


--
-- TOC entry 3798 (class 0 OID 0)
-- Dependencies: 206
-- Name: TABLE cp_features; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cp_features TO PUBLIC;


--
-- TOC entry 3800 (class 0 OID 0)
-- Dependencies: 207
-- Name: SEQUENCE cp_features_feature_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cp_features_feature_id_seq TO PUBLIC;


--
-- TOC entry 3801 (class 0 OID 0)
-- Dependencies: 208
-- Name: TABLE cp_groups; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cp_groups TO PUBLIC;


--
-- TOC entry 3803 (class 0 OID 0)
-- Dependencies: 209
-- Name: SEQUENCE cp_groups_group_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cp_groups_group_id_seq TO PUBLIC;


--
-- TOC entry 3804 (class 0 OID 0)
-- Dependencies: 210
-- Name: TABLE cu_alert; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_alert TO PUBLIC;


--
-- TOC entry 3805 (class 0 OID 0)
-- Dependencies: 211
-- Name: TABLE cu_alert_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_alert_audit TO PUBLIC;


--
-- TOC entry 3807 (class 0 OID 0)
-- Dependencies: 212
-- Name: SEQUENCE cu_alert_cu_alert_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_alert_cu_alert_id_seq TO PUBLIC;


--
-- TOC entry 3808 (class 0 OID 0)
-- Dependencies: 213
-- Name: TABLE cu_alert_publishinfo; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_alert_publishinfo TO PUBLIC;


--
-- TOC entry 3810 (class 0 OID 0)
-- Dependencies: 214
-- Name: SEQUENCE cu_alert_publishinfo_cu_alert_publishinfo_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_alert_publishinfo_cu_alert_publishinfo_id_seq TO PUBLIC;


--
-- TOC entry 3811 (class 0 OID 0)
-- Dependencies: 215
-- Name: SEQUENCE cu_alert_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.cu_alert_seq TO PUBLIC;


--
-- TOC entry 3812 (class 0 OID 0)
-- Dependencies: 216
-- Name: TABLE cu_connection_access; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_connection_access TO PUBLIC;


--
-- TOC entry 3813 (class 0 OID 0)
-- Dependencies: 217
-- Name: TABLE cu_connection_access_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_connection_access_audit TO PUBLIC;


--
-- TOC entry 3815 (class 0 OID 0)
-- Dependencies: 218
-- Name: SEQUENCE cu_connection_access_connection_access_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_connection_access_connection_access_id_seq TO PUBLIC;


--
-- TOC entry 3816 (class 0 OID 0)
-- Dependencies: 219
-- Name: TABLE cu_connection_types; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_connection_types TO PUBLIC;


--
-- TOC entry 3818 (class 0 OID 0)
-- Dependencies: 220
-- Name: SEQUENCE cu_connection_types_type_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_connection_types_type_id_seq TO PUBLIC;


--
-- TOC entry 3819 (class 0 OID 0)
-- Dependencies: 221
-- Name: TABLE cu_connections; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_connections TO PUBLIC;


--
-- TOC entry 3821 (class 0 OID 0)
-- Dependencies: 222
-- Name: SEQUENCE cu_connections_connections_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_connections_connections_id_seq TO PUBLIC;


--
-- TOC entry 3822 (class 0 OID 0)
-- Dependencies: 223
-- Name: TABLE cu_dashboard; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_dashboard TO PUBLIC;


--
-- TOC entry 3824 (class 0 OID 0)
-- Dependencies: 224
-- Name: SEQUENCE cu_dashboard_cu_dashboard_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_dashboard_cu_dashboard_id_seq TO PUBLIC;


--
-- TOC entry 3825 (class 0 OID 0)
-- Dependencies: 225
-- Name: SEQUENCE cu_hash_criteria_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.cu_hash_criteria_id_seq TO PUBLIC;


--
-- TOC entry 3826 (class 0 OID 0)
-- Dependencies: 226
-- Name: TABLE cu_hash_criteria; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_hash_criteria TO PUBLIC;


--
-- TOC entry 3827 (class 0 OID 0)
-- Dependencies: 227
-- Name: SEQUENCE cu_report_visualization_cu_report_visualization_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.cu_report_visualization_cu_report_visualization_id_seq TO PUBLIC;


--
-- TOC entry 3828 (class 0 OID 0)
-- Dependencies: 228
-- Name: TABLE cu_report_visualization; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_report_visualization TO PUBLIC;


--
-- TOC entry 3829 (class 0 OID 0)
-- Dependencies: 229
-- Name: TABLE cu_report_visualization_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_report_visualization_audit TO PUBLIC;


--
-- TOC entry 3830 (class 0 OID 0)
-- Dependencies: 230
-- Name: TABLE cu_schedule; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_schedule TO PUBLIC;


--
-- TOC entry 3832 (class 0 OID 0)
-- Dependencies: 231
-- Name: SEQUENCE cu_schedule_cu_schedule_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_schedule_cu_schedule_id_seq TO PUBLIC;


--
-- TOC entry 3833 (class 0 OID 0)
-- Dependencies: 232
-- Name: TABLE cu_schedule_log; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_schedule_log TO PUBLIC;


--
-- TOC entry 3835 (class 0 OID 0)
-- Dependencies: 233
-- Name: SEQUENCE cu_schedule_log_cu_schedule_log_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_schedule_log_cu_schedule_log_id_seq TO PUBLIC;


--
-- TOC entry 3836 (class 0 OID 0)
-- Dependencies: 234
-- Name: TABLE cu_shared_connection_access; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_shared_connection_access TO PUBLIC;


--
-- TOC entry 3838 (class 0 OID 0)
-- Dependencies: 235
-- Name: SEQUENCE cu_shared_connection_access_cu_shared_connection_access_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_shared_connection_access_cu_shared_connection_access_id_seq TO PUBLIC;


--
-- TOC entry 3839 (class 0 OID 0)
-- Dependencies: 236
-- Name: SEQUENCE cu_shared_model_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.cu_shared_model_id_seq TO PUBLIC;


--
-- TOC entry 3840 (class 0 OID 0)
-- Dependencies: 237
-- Name: TABLE cu_shared_model; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_shared_model TO PUBLIC;


--
-- TOC entry 3841 (class 0 OID 0)
-- Dependencies: 238
-- Name: TABLE cu_shared_model_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_shared_model_audit TO PUBLIC;


--
-- TOC entry 3842 (class 0 OID 0)
-- Dependencies: 239
-- Name: TABLE cu_shared_visualization; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_shared_visualization TO PUBLIC;


--
-- TOC entry 3844 (class 0 OID 0)
-- Dependencies: 240
-- Name: SEQUENCE cu_shared_visualization_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_shared_visualization_id_seq TO PUBLIC;


--
-- TOC entry 3845 (class 0 OID 0)
-- Dependencies: 241
-- Name: TABLE cu_storybook_visualization; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_storybook_visualization TO PUBLIC;


--
-- TOC entry 3847 (class 0 OID 0)
-- Dependencies: 242
-- Name: SEQUENCE cu_storybook_visualization_storybook_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.cu_storybook_visualization_storybook_id_seq TO PUBLIC;


--
-- TOC entry 3848 (class 0 OID 0)
-- Dependencies: 243
-- Name: TABLE cu_user_groups; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cu_user_groups TO PUBLIC;


--
-- TOC entry 3850 (class 0 OID 0)
-- Dependencies: 244
-- Name: SEQUENCE cu_user_groups_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.cu_user_groups_id_seq TO PUBLIC;


--
-- TOC entry 3851 (class 0 OID 0)
-- Dependencies: 245
-- Name: SEQUENCE data_entity_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.data_entity_id TO PUBLIC;


--
-- TOC entry 3852 (class 0 OID 0)
-- Dependencies: 246
-- Name: TABLE data_entity; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_entity TO PUBLIC;


--
-- TOC entry 3853 (class 0 OID 0)
-- Dependencies: 247
-- Name: SEQUENCE data_hub_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.data_hub_id TO PUBLIC;


--
-- TOC entry 3854 (class 0 OID 0)
-- Dependencies: 248
-- Name: TABLE data_hub; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_hub TO PUBLIC;


--
-- TOC entry 3855 (class 0 OID 0)
-- Dependencies: 249
-- Name: SEQUENCE data_hub_entity_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.data_hub_entity_id TO PUBLIC;


--
-- TOC entry 3856 (class 0 OID 0)
-- Dependencies: 250
-- Name: TABLE data_hub_entity; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_hub_entity TO PUBLIC;


--
-- TOC entry 3857 (class 0 OID 0)
-- Dependencies: 251
-- Name: TABLE data_hub_entity_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_hub_entity_audit TO PUBLIC;


--
-- TOC entry 3858 (class 0 OID 0)
-- Dependencies: 252
-- Name: SEQUENCE data_model_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.data_model_id TO PUBLIC;


--
-- TOC entry 3859 (class 0 OID 0)
-- Dependencies: 253
-- Name: TABLE data_model; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_model TO PUBLIC;


--
-- TOC entry 3860 (class 0 OID 0)
-- Dependencies: 254
-- Name: SEQUENCE data_model_entity_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.data_model_entity_id TO PUBLIC;


--
-- TOC entry 3861 (class 0 OID 0)
-- Dependencies: 255
-- Name: TABLE data_model_entity; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_model_entity TO PUBLIC;


--
-- TOC entry 3862 (class 0 OID 0)
-- Dependencies: 256
-- Name: TABLE data_model_entity_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_model_entity_audit TO PUBLIC;


--
-- TOC entry 3863 (class 0 OID 0)
-- Dependencies: 257
-- Name: SEQUENCE data_model_entity_relation_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.data_model_entity_relation_id TO PUBLIC;


--
-- TOC entry 3864 (class 0 OID 0)
-- Dependencies: 258
-- Name: TABLE data_model_entity_relation; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_model_entity_relation TO PUBLIC;


--
-- TOC entry 3865 (class 0 OID 0)
-- Dependencies: 259
-- Name: TABLE data_model_entity_relation_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_model_entity_relation_audit TO PUBLIC;


--
-- TOC entry 3866 (class 0 OID 0)
-- Dependencies: 260
-- Name: SEQUENCE data_model_mf_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.data_model_mf_id TO PUBLIC;


--
-- TOC entry 3867 (class 0 OID 0)
-- Dependencies: 261
-- Name: TABLE data_model_multifact; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.data_model_multifact TO PUBLIC;


--
-- TOC entry 3868 (class 0 OID 0)
-- Dependencies: 262
-- Name: SEQUENCE es_temp_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.es_temp_id_seq TO PUBLIC;


--
-- TOC entry 3869 (class 0 OID 0)
-- Dependencies: 263
-- Name: TABLE es_temp; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.es_temp TO PUBLIC;


--
-- TOC entry 3870 (class 0 OID 0)
-- Dependencies: 264
-- Name: SEQUENCE export_temp_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.export_temp_seq TO PUBLIC;


--
-- TOC entry 3871 (class 0 OID 0)
-- Dependencies: 265
-- Name: TABLE export_temp; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.export_temp TO PUBLIC;


--
-- TOC entry 3872 (class 0 OID 0)
-- Dependencies: 266
-- Name: TABLE ldap_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.ldap_audit TO PUBLIC;


--
-- TOC entry 3873 (class 0 OID 0)
-- Dependencies: 267
-- Name: TABLE ldap_configurations; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.ldap_configurations TO PUBLIC;


--
-- TOC entry 3875 (class 0 OID 0)
-- Dependencies: 268
-- Name: SEQUENCE ldap_configurations_ldap_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.ldap_configurations_ldap_id_seq TO PUBLIC;


--
-- TOC entry 3876 (class 0 OID 0)
-- Dependencies: 269
-- Name: TABLE log_patterns; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.log_patterns TO PUBLIC;


--
-- TOC entry 3878 (class 0 OID 0)
-- Dependencies: 270
-- Name: SEQUENCE log_patterns_log_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.log_patterns_log_id_seq TO PUBLIC;


--
-- TOC entry 3879 (class 0 OID 0)
-- Dependencies: 271
-- Name: TABLE login_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.login_audit TO PUBLIC;


--
-- TOC entry 3880 (class 0 OID 0)
-- Dependencies: 272
-- Name: TABLE mail_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.mail_audit TO PUBLIC;


--
-- TOC entry 3882 (class 0 OID 0)
-- Dependencies: 273
-- Name: SEQUENCE mail_audit_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.mail_audit_id_seq TO PUBLIC;


--
-- TOC entry 3883 (class 0 OID 0)
-- Dependencies: 274
-- Name: TABLE ml_models; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.ml_models TO PUBLIC;


--
-- TOC entry 3884 (class 0 OID 0)
-- Dependencies: 275
-- Name: SEQUENCE ml_temp_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.ml_temp_id_seq TO PUBLIC;


--
-- TOC entry 3885 (class 0 OID 0)
-- Dependencies: 276
-- Name: TABLE ml_temp; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.ml_temp TO PUBLIC;


--
-- TOC entry 3886 (class 0 OID 0)
-- Dependencies: 277
-- Name: SEQUENCE password_policy_rules_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.password_policy_rules_id_seq TO PUBLIC;


--
-- TOC entry 3887 (class 0 OID 0)
-- Dependencies: 278
-- Name: TABLE password_policy_rules; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.password_policy_rules TO PUBLIC;


--
-- TOC entry 3888 (class 0 OID 0)
-- Dependencies: 279
-- Name: TABLE securityquestions_user_registration; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.securityquestions_user_registration TO PUBLIC;


--
-- TOC entry 3889 (class 0 OID 0)
-- Dependencies: 280
-- Name: TABLE semantic_names; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.semantic_names TO PUBLIC;


--
-- TOC entry 3891 (class 0 OID 0)
-- Dependencies: 281
-- Name: SEQUENCE semantic_names_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.semantic_names_id_seq TO PUBLIC;


--
-- TOC entry 3892 (class 0 OID 0)
-- Dependencies: 295
-- Name: TABLE smartinsight_temp; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.smartinsight_temp TO PUBLIC;


--
-- TOC entry 3894 (class 0 OID 0)
-- Dependencies: 294
-- Name: SEQUENCE smartinsight_temp_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.smartinsight_temp_id_seq TO PUBLIC;


--
-- TOC entry 3895 (class 0 OID 0)
-- Dependencies: 282
-- Name: TABLE storage_views; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.storage_views TO PUBLIC;


--
-- TOC entry 3896 (class 0 OID 0)
-- Dependencies: 283
-- Name: TABLE user_account; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.user_account TO PUBLIC;


--
-- TOC entry 3897 (class 0 OID 0)
-- Dependencies: 284
-- Name: TABLE user_account_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.user_account_audit TO PUBLIC;


--
-- TOC entry 3899 (class 0 OID 0)
-- Dependencies: 285
-- Name: SEQUENCE user_account_user_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.user_account_user_id_seq TO PUBLIC;


--
-- TOC entry 3900 (class 0 OID 0)
-- Dependencies: 286
-- Name: TABLE user_role; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.user_role TO PUBLIC;


--
-- TOC entry 3901 (class 0 OID 0)
-- Dependencies: 287
-- Name: TABLE user_role_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.user_role_audit TO PUBLIC;


--
-- TOC entry 3903 (class 0 OID 0)
-- Dependencies: 288
-- Name: SEQUENCE user_role_role_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.user_role_role_id_seq TO PUBLIC;


--
-- TOC entry 3904 (class 0 OID 0)
-- Dependencies: 289
-- Name: SEQUENCE workspace_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.workspace_id TO PUBLIC;


--
-- TOC entry 3905 (class 0 OID 0)
-- Dependencies: 290
-- Name: TABLE workspace; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.workspace TO PUBLIC;


--
-- TOC entry 3906 (class 0 OID 0)
-- Dependencies: 291
-- Name: SEQUENCE workspace_entity_id; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.workspace_entity_id TO PUBLIC;


--
-- TOC entry 3907 (class 0 OID 0)
-- Dependencies: 292
-- Name: TABLE workspace_entity; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.workspace_entity TO PUBLIC;


--
-- TOC entry 3908 (class 0 OID 0)
-- Dependencies: 293
-- Name: TABLE workspace_entity_audit; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.workspace_entity_audit TO PUBLIC;


--
-- TOC entry 2061 (class 826 OID 24583)
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON SEQUENCES  FROM postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT,USAGE ON SEQUENCES  TO PUBLIC;


--
-- TOC entry 2062 (class 826 OID 24584)
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON TABLES  FROM postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES  TO PUBLIC;


-- Completed on 2020-06-19 09:59:47

--
-- PostgreSQL database dump complete
--

